# RelTree：SOS → Room → SPLIT / OBJECT

版本：`room_root_ar_v3`，2026-09-28。边界是 condition；Room 是要生成的根节点。有条件与无条件采样都由模型生成 Room 尺寸，再生成物体布局，不生成边界多边形。

## SOS 在同一个 self-attention 序列中

```text
主干输入序列（逻辑示意，数组位置不参与位置编码）：

  [boundary/type condition tokens] [SOS] [C_Room] [C_A] [C_B] ... [X_A] [X_B] ...
  └──────────────── 同一个主干 Transformer / masked self-attention ──────────────┘
                                     │                         │
                                   h_SOS                     h_A, h_B, ...
                                     │                         │
                         两层因果属性 Transformer     同一组两层因果属性 Transformer
                                     │                         │
                           Room.sx → sy → sz            type → class → geometry
```

SOS 是可训练 embedding，通过同一套主干 attention 与条件 token 交互，其输出 hidden state 预测 Room。没有模型外预先读取边界再计算 Room 的分支，也没有使用 GT room frame 的 `room_encoder`。`root_sos` 与每层第一个节点用于 shifted 输入的 `node_sos` 是不同参数。

条件序列含房间类型 token 与沿多边形周长采样的 32 个边界点/法向 token。边界坐标以固定 **16 m** 标度归一化，保留米制大小信息；不除以待预测的 GT Room 尺寸。条件本身可提供房间大小信息，这是已知输入。无条件时全部条件槽替换成学习到的 null embedding。

训练保留独立 GT context 流。SOS 是 depth=0 的预测 query；GT Room 是 depth=0 的 context，二者不是同一个 token。物理打包将 SOS 放在尾部，逻辑上它是第一个预测位置；没有使用数组绝对下标。所有层应用相同 mask。

`1` 表示可见，`.` 表示遮挡；为简洁合并所有边界条件列：

```text
读方\key  condition | SOS C_Room C_A C_B C_C C_D C_E C_F | X_A X_B X_C X_D X_E X_F
condition     1    |  .    .     .   .   .   .   .   .  |  .   .   .   .   .   .
SOS           1    |  1    .     .   .   .   .   .   .  |  .   .   .   .   .   .
C_Room        1    |  .    1     .   .   .   .   .   .  |  .   .   .   .   .   .
C_A / C_B     1    |  .    1     1   1   .   .   .   .  |  .   .   .   .   .   .
X_A           1    |  .    1     .   .   .   .   .   .  |  1   .   .   .   .   .
X_B           1    |  .    1     .   .   .   .   .   .  |  1   1   .   .   .   .
X_C           1    |  .    1     1   1   .   .   .   .  |  .   .   1   .   .   .
X_D           1    |  .    1     1   1   .   .   .   .  |  .   .   1   1   .   .
X_E           1    |  .    1     1   1   .   .   .   .  |  .   .   1   1   1   .
X_F           1    |  .    1     1   1   .   .   .   .  |  .   .   1   1   1   1
```

SOS 不读 GT Room。后续节点使用已经生成或 teacher-forced 的 Room context，而不是预测 Room 时的 query hidden；这沿用布局阶段的 context/query 双流协议。推理结束 Room 阶段后，可以移除 SOS 预测槽，与以上遮挡规则一致。条件始终通过同一个 self-attention 对后续节点可见。

## Room 的免训练 token

Room 没有父节点，因此使用三个**绝对完整边长** bin，顺序为 x、y、z；后续 SPLIT/OBJECT 沿用原有 7 维父子相对 token。Room 类型在首位置固定，不预测类别、位置或 yaw。

| 属性 | 固定范围 | bin 数 | bin 宽度 |
| --- | --- | --- | --- |
| sx | (0, 16] m | 256 | 0.0625 m |
| sy（高度） | (0, 8] m | 256 | 0.03125 m |
| sz | (0, 16] m | 256 | 0.0625 m |

均匀量化、bin 中点解码，超范围训练目标直接报错。未发生预测错误时，仅根尺寸量化的逐轴误差上限为半个 bin；不能据此推断完整布局误差，后者还包含相对量化和完整父链传播。

固定 yaw=0，水平中心 x=z=0、地面 y=0，因此 Room 中心为 `(0, sy/2, 0)`。这是规范坐标原点，不是额外预测的位置属性。当前修正数据的高度由预处理统一设为 3 m，模型暂时只能学习这个固定高度，不能据此评价真实房高分布。

Root 属性分支：

```text
输入：project(h_SOS), Emb(sx_GT), Emb(sy_GT)
目标：      sx,            sy,          sz
```

使用两层 causal Transformer，与布局属性分支共享 Transformer blocks、投影及最终 norm，独立使用 Room 尺寸 embedding、位置 embedding 和三个分类 head。采样时前缀换成刚采样的尺寸。主干不读取这个当前 Room 属性前缀。

概率分解为：

```text
p(Room, T | boundary/type 或 null)
 = p(sx | h_SOS) p(sy | h_SOS, sx) p(sz | h_SOS, sx, sy)
   × ∏_level ∏_node p(node | Room, earlier levels, preceding nodes, condition)
```

Room 仅有三个尺寸 CE；布局节点的类型/类别/相对几何监督不变。默认将 Room 作为一个有效 query，与布局 query 一起按场景平均，再按 batch 平均。`room_size_weight` 可调整其权重，日志单列 `room_size_loss`，其值是对总损失的场景归一化贡献。

## 生成与数据

生成先运行含 condition 和 SOS 的主干序列，再串行采样 Room 的三个尺寸。预测尺寸 token 进入根 context，解码出的 Room 为直接子节点提供父框；后续所有节点按照已经生成的父框递推。

单场景 `generate` 和批量 `generate_batch` 使用相同规则。真实多边形始终保留为条件。

Python API（在 Slurm GPU 作业执行的 `.sh` 脚本中调用）：

```python
from networks.reltree.training import load_model
from networks.reltree.room_root import BoundaryCondition

model, tokenizer, checkpoint = load_model('path/to/v3/last.pt', 'cuda:0')

# 提供边界和房间类型；Room 尺寸由模型预测。
condition = BoundaryCondition(boundary=[[-2, -3], [2, -3], [2, 3], [-2, 3]])
conditional = model.generate(condition, tokenizer, seed=42)

# 完全不提供外部 frame、边界、房间类型。
unconditional = model.generate(tokenizer=tokenizer, seed=42)
print(unconditional.room, unconditional.objects, unconditional.status)
```

也支持 `SceneCondition`，采样忽略其 `.room`，仅使用边界和房间类型。`unconditional=True` 时忽略整个条件对象。v3 输出 `room`、`room_size_codes` 表示预测的根框与历史 token。`condition_boundary` 仅记录已给定条件，无条件为 null。根框的矩形足迹不代表墙体多边形。

本项目 room-conditioned 评测脚本使用边界和 livingroom 类型作为输入，不提供 GT Room 框或家具 token。默认结果目录为 `reltree_last`；协议校验拒绝在协议或源码改变后混合续采。历史 GT 根框实验的结果保留在 `reltree_last_gt_room`。

修正后的 `datasets/processed_reltree_bedroom_v2` 可直接复用：collator 从已有 `encoded.room` 构造新增监督。原 layout token 不变，GT Root context 的尺寸也先量化，以匹配推理时的输入。`tokenizer.decode(result.encoded)` 使用生成的 Room，按完整链解码。

代码：`room_root.py` 负责量化和打包；`root_model.py` 负责 SOS、主干与属性预测；`root_generation.py` 负责先生成 Room、再调用共享布局采样器。训练入口根据 `architecture` 分派。检查点保存并核验独立 `room_tokenizer` 协议，运行目录还保存 `room_tokenizer.json`。

v2 已训练权重仍可加载，但不能作为 v3 的断点继续训练；v3 需要重新训练。旧采样画图脚本中的“external frame retained”说明针对 v2，v3 的结果以新生成 API 为准。

## 训练入口

`training_configs/reltree_root_ar.yaml`：12×768、FFN×4，属性分支 2×256、FFN×4，cond_drop=10%，batch=128、BF16、100k steps。输出独立目录 `runs/bedroom_root_ar_12x768_attr256_cd10`，参数量 **91,899,937**。

```bash
# 完整机制测试，包括真实 signal 中断恢复。
sbatch networks/reltree/verify_resume.slurm

# 新架构 GPU 短训、全尺寸配置恢复、单场景条件/无条件自由生成验证。
sbatch networks/reltree/validate_root_ar.slurm

# 全量训练；已有此架构的 last.pt 时自动恢复。
sbatch networks/reltree/train_root_reltree.slurm
```

全部 Python 命令放在 `.sh` 中，通过 Slurm 分配单张 GPU。全量作业时限 48 小时，到时前在安全 optimizer 边界存盘；重新执行同一 `sbatch` 命令即可从 `last.pt` 恢复，单独这个 Slurm 入口不自动重提。不会覆盖 v2 的运行目录。

2026-09-28：作业 **16329** 在 A800 上通过全部 **90 项测试**，包括 SOS/GT Root 隔离、固定边界标度、主干 attention 条件依赖、Room 属性因果性、一次 forward 与逐节点/属性 logits 一致性、padding/打包顺序、无外部 frame 解码及含条件丢弃的中断恢复。日志：`slurm/reltree-resume-test-16329.out`。

GPU 集成作业 **16330** 同样 `COMPLETED / 0:0`：

| 验证 | 结果 |
| --- | --- |
| 完整 12×768 配置、128 train / 16 val、BF16、batch 128 | 24 步，train CE 37.5566 → 22.5837；val CE 37.5045 → 24.3351 |
| 完整模型恢复 | 从 24 步续到 26 步；模型、optimizer、Room codec 正常加载 |
| 该短训显存 | 首段峰值 allocated 4936.25 MiB；含恢复过程峰值 5291.77 MiB，reserved 6172 MiB；不是全量数据最大场景保证值 |
| 小模型单场景过拟合 | 400 步，train CE 39.7466 → 0.001266，恢复到 405 步 |
| 条件自由生成 | Room 的三个 bin 与全部布局 token 精确复现；11 次节点调用、85 次属性决策 |
| 无条件自由生成 | 没有传入条件/frame，Room 与布局 token 同样精确复现 |

本次贪心采样记录 CUDA 同步耗时，条件 0.16792 s，无条件 0.15730 s；各只有一个过拟合样本且没有独立预热，不能作为性能比较。单场景验证的 val CE 上升至约 64，只用于检查可学习性及自由生成链路，不能代表泛化质量。

完整结果：[GPU 验证汇总](runs/validation-root-ar-16330/validation_summary.json)、[条件及无条件输出](runs/validation-root-ar-16330/overfit_generation.json)、[短训日志](slurm/reltree-root-check-16330.out)。

## 2026-09-28 全量训练启动

按用户要求于 11:16（Asia/Shanghai）提交作业 **16331**，Slurm 已分配单 GPU，状态 `RUNNING`。使用上述 12×768 新架构配置，从头训练至累计 100,000 steps；复用修正后的完整 train/val 数据。运行目录为 `runs/bedroom_root_ar_12x768_attr256_cd10`。

持久 tmux 会话 `reltree-root-12x768` 运行 `run_root_pipeline.sh`；该包装脚本复用已经通过的 16329 机制测试和已有数据，通过 `sbatch --wait` 提交训练。超时、抢占或 USR1 安全存盘后退出 75 时，流水线重新提交，并自动从 `last.pt` 恢复。普通代码错误或人工取消会停止流水线，不盲目重试。每次作业申请 48 小时。

```bash
tmux attach -t reltree-root-12x768
tail -f networks/reltree/slurm/reltree-root-12x768-16331.out
```

当前作业编号：[train_job.txt](runs/bedroom_root_ar_12x768_attr256_cd10/train_job.txt)；提交及恢复记录：[pipeline.log](runs/bedroom_root_ar_12x768_attr256_cd10/pipeline.log)；训练指标：[metrics.jsonl](runs/bedroom_root_ar_12x768_attr256_cd10/metrics.jsonl)。
