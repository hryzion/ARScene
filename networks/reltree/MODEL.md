# RelTree：逐层节点自回归与属性自回归

2026-09-28 新增 **`room_root_ar_v3`**：SOS 加入主干 self-attention 序列，预测 Room 长宽高；边界作为独立条件。无条件生成无需外部 room frame。实现、mask、量化协议和新训练入口见 [ROOT_AR.md](ROOT_AR.md)。

**以下记录 `node_attribute_ar_v2`（2026-09-26）的设计和历史训练，不代表 v3 的房间生成协议。** v2 已知房间类型、边界、room frame，逐层生成 SPLIT/OBJECT；每层内部按路径顺序串行生成节点，每个节点内部再串行生成属性。tokenizer 保持免训练，Transformer、embedding 和 head 正常监督训练；覆盖 Layout 阶段。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `model.py` | query/shifted 节点融合、主干 Transformer、两层因果属性 Transformer |
| `packing.py` | 条件、GT context、预测 query 的打包、树位置编码输入、attention 可见性 |
| `loss.py` | 仅 query 的类型、几何、类别损失；按场景平均 |
| `generation.py` | 动态分配 query，逐层、逐节点、逐属性串行采样 |
| `prepare.py` | 读取根目录 `preprocess.py` 生成的正确 OBB NPZ，构树与量化 |
| `data.py` | `RelTreeDataset`、`RelTreeCollator` |
| `training.py` / 根目录 `train_reltree.py` | 训练、验证、checkpoint、恢复、生成失败统计 |

默认主干：width=256、6 层、8 heads、FFN 倍数 4、最大深度 32、32 个边界采样点。属性 decoder：width=128、2 层、4 heads、FFN 倍数 2。31 类、当前 bin 配置下共 **6,240,033** 个参数，其中属性 decoder 为 **736,545** 个。房间由一个 frame/type token 和边界点/法向 token 表示。条件几何来自房间本身，不从物体数量或未来布局推断。

当前全量训练配置 `training_configs/reltree_full.yaml` 使用 **8 层、512 维、8 heads、cond_drop=0.1**，属性 decoder 仍为两层 128 维，共 **28,043,041** 个参数。

`cond_drop` 在训练时按场景独立采样：以 10% 概率把整个房间条件序列（类型、frame、边界）替换为共享的可学习空条件 embedding。同一场景的全部尺度使用同一个丢弃决定，保留树历史和 shifted 节点；不改变节点/属性的因果 mask，也不删除条件槽位。验证和常规生成不随机丢条件。外部坐标解码仍使用已知 room frame，因此该机制不等于生成房间条件的完全无条件场景模型。普通网络 dropout 独立保留为 0.1。

## Teacher forcing 与位置特征

一次 forward 包含所有尺度的 GT context 和预测 query。物理打包顺序是 `conditions + contexts + queries + padding`，该数组下标不参与位置编码。

每个 context 输入自己的类型、量化几何和可选类别。原始 query 由父节点的已知属性和自己的 role/depth/path/slot 构成。在**特征维**拼接同层前一个完整节点的 raw embedding，再线性投影回主干宽度：

```text
x(l,i) = Linear(concat(query(l,i), Emb(z(l,i-1))))
h(l,i) = Backbone(x(l,i), legal_history)
```

每层第一个节点用可训练的 `node_sos` 代替前序节点；层内按树路径字典序排列，跨父节点继续 shift，每层重新从 SOS 开始。`Emb(z)` 是类型、类别、7 维几何的原始 embedding 之和，不能取经过 attention 的 GT context hidden。query 不输入自己或未来节点的完整目标，也不输入资产 ID、最终树深度。训练时前序节点取 GT，采样时取已生成节点。

以下规则在每一层 self-attention 都执行：

| 读取方 | 可读 key |
| --- | --- |
| 条件 | 所有条件 token |
| C_i | 条件与深度 ≤ i 的 context，不读 query |
| Q_(l,i) | 条件、深度 < l 的所有 context、同深度且顺序 ≤ i 的 query |
| padding | 不是有效 key，无监督；其 query 行允许读条件以避免全遮挡 softmax |

位置由角色、深度、左右槽与完整二进制路径编码。所有历史尺度保留，浅层 OBJECT 也保留。当前层原始 context 整体遮挡；同层前序 GT 只通过 shifted query 输入。

以 `Room → (A,B)`、`A → (C,D)`、`B → (E,F)` 为例，融合后的 query 输入依次为：

```text
X_A = concat(Q_A, SOS)     X_B = concat(Q_B, Emb(A))
X_C = concat(Q_C, SOS)     X_D = concat(Q_D, Emb(C))
X_E = concat(Q_E, Emb(D))  X_F = concat(Q_F, Emb(E))
```

下面只画 query 行。`1` 可读，`.` 遮挡；所有行还可读房间条件。`C_*` 表示原始 GT context，`X_*` 表示融合后的 query token。

```text
读方\key  C_Room C_A C_B C_C C_D C_E C_F | X_A X_B | X_C X_D X_E X_F
X_A          1    .   .   .   .   .   .  |  1   .  |  .   .   .   .
X_B          1    .   .   .   .   .   .  |  1   1  |  .   .   .   .
X_C          1    1   1   .   .   .   .  |  .   .  |  1   .   .   .
X_D          1    1   1   .   .   .   .  |  .   .  |  1   1   .   .
X_E          1    1   1   .   .   .   .  |  .   .  |  1   1   1   .
X_F          1    1   1   .   .   .   .  |  .   .  |  1   1   1   1
```

这就是每层的“历史平台＋同层阶梯”。例如 X_E 可以经自身的 shifted 输入读取 D，但不能读 C_E 或 X_F 中的 E。历史 context 不读未来 context 或任何 query，因此多层 attention 也不能绕过遮挡。

`pack_teacher_forcing(..., query_depth=l)` 仅用于逐尺度 GT-history 对照：删除当前/未来 context，重新打包相同的合法历史和当前 query。它不用于推理预测查询数量。

## 两层属性 decoder、监督和根协议

每个 `h(l,i)` 作为独立属性序列的 SOS，经线性投影到 128 维。所有节点共享同一个两层因果 Transformer，节点间不在此 decoder 中互相 attention。

```text
输入位置：project(h_i), Emb(type), Emb(class), Emb(x), Emb(y), Emb(z), Emb(yaw), Emb(sx), Emb(sy)
预测目标：    type,        class,      x,        y,      z,      yaw,      sx,      sy,      sz
```

这里 `sx/sy/sz` 都指 log 尺寸比的离散 bin。每个输出位置有对应属性的 head；其 hidden 已读取前面的属性，所以 head 分开不等于属性独立采样。训练时用 shifted GT 属性，推理时用真实采样的属性前缀。主干完全不读取 `attribute_prefix`，因此当前类型不会泄漏进预测它的 `h_i`。

模型的因子分解为：

```text
p(T | c) = ∏_l ∏_i p(z(l,i) | F_<l, z(l,<i), c)
p(z(l,i) | ...) = ∏_a p(z(l,i,a) | h(l,i), z(l,i,<a))
```

它显式建模同层节点与节点内属性的联合分布；是否减少重复、碰撞和越界仍需实验，结构本身不保证这些性质。

- 所有尺度、节点、属性在一次 teacher-forcing forward 中计算。属性 decoder 只为有效 query 构造序列。
- OBJECT 监督类型、全部几何和类别；SPLIT 监督类型和几何，但沿用 tokenizer 的固定相对 yaw=0 协议，yaw 无 loss。
- EMPTY 只监督类型，且只允许单物体根的右槽。
- context、condition、padding 无监督；没有类别目标时类别 loss 为零。
- 几何 head 可读取当前节点已在前缀中的类型与类别；训练和推理的条件一致。
- 对有效 query 的加权损失先按场景平均，再按 batch 平均。

根的类型使用自回归约束。左槽只允许 SPLIT/OBJECT；生成右槽时，只有左槽已为 OBJECT 才允许 EMPTY。仍只有五种合法组合：

```text
(SPLIT, SPLIT), (SPLIT, OBJECT), (OBJECT, SPLIT),
(OBJECT, OBJECT), (OBJECT, EMPTY)
```

训练对每个 query 的条件类型分布分别做 CE，应用相同的合法性 mask；不再使用旧版五种组合 logit 相加的联合 head。右槽的预测依赖完整左节点，其他深度只允许 SPLIT/OBJECT。SPLIT 的类别用缺省标记、yaw 用固定零 bin 进入后续属性前缀；这两项不采样也无 loss。EMPTY 在 type 后立即结束。

## 动态推理

```python
from networks.reltree.model import RelTree, RelTreeConfig
from networks.reltree.packing import SceneCondition
from networks.reltree.training import load_model

model, tokenizer, checkpoint = load_model("path/to/last.pt", device="cuda:0")
result = model.generate(condition, tokenizer, temperature=0., max_depth=16, max_objects=64)
objects = result.objects
print(result.status)
```

`condition` 只含已知 room frame、room type 和边界。先为 Room 建立两个槽，之后每个已经采样的 SPLIT 恰产生两个槽。生成出的 OBJECT 保留在历史与结果中，不再展开。几何递推使用已生成父框，绝不读取 GT 物体数量、GT 当层类型或 GT 最终深度。

对于训练时启用了 `cond_drop > 0` 的当前自回归模型，`model.generate(..., unconditional=True)` 会在每个节点预测时用 learned null condition 替换全部房间条件 embedding，保留已生成历史。外部 `condition.room` 仍用于坐标解码，因此这不是连同房间边界一起生成的完全无条件模型。默认 `unconditional=False` 保留房间条件；没有训练 null condition 的检查点拒绝无条件采样。

固定一份 `last.pt` 快照后，可运行 `sbatch networks/reltree/sample_rooms.slurm --checkpoint /path/to/snapshot.pt --output /path/to/new-output`，从验证集固定随机选择 4 个房间，分别采样 conditional / null-condition 两种模式，保存原始 token、布局 JSON、诊断、俯视图与 3D OBB 图。默认 temperature=1、top_k=0；两种模式每对使用相同 seed，不按结果质量筛选，GT 家具仅用于绘图对照。脚本先验证条件屏蔽和串行采样，再执行采样。

已完成训练的 `best.pt` / `last.pt` 可用 `sample_comparison.slurm` 在同一批房间、相同对应 seed 上比较。例如：

```bash
sbatch networks/reltree/sample_comparison.slurm \
  --run networks/reltree/runs/bedroom_ar_12x768_attr256_cd10 \
  --output networks/reltree/runs/bedroom_ar_12x768_attr256_cd10/samples_best_last_16_20260927 \
  --count 16 --seed 20260927 --temperature 1 --top-k 0
```

该设置共保存 64 个正式样本，按 4 个房间一页输出 GT 与四组生成结果的俯视/3D 对比图，`index.html` 提供离线图集。`sampling_times.csv` 保存每个场景的 GPU 同步 wall-clock 时间、物体数、节点/属性采样次数及结构状态；`report.json` 汇总每组总耗时、均值、中位数、P95、检查点哈希和所有原始输出。计时包含 `generate` 内部输入打包、传输、串行预测和几何解码，不包含模型加载、预热、诊断、文件输出及绘图；每个检查点对两种模式各预热一次并单独记录。推理精度为 FP32，单场景顺序执行，各组包含全部请求，失败样本也保留。样本数量与树结构不同会影响耗时，不能把组间时间差直接当作条件注入的固定成本。

超过深度/节点/物体上限、尺寸非法或采样分布异常时，返回明确失败状态和已有部分输出，不强行把 SPLIT 转成 OBJECT。训练结束的 `samples.json` 保留成功与失败的全部请求，`summary.json` 统计各状态。`complete` 只表示树展开完成，不代表家具分布、碰撞、支撑或越界质量达标。

每层固定较早层的 history，按路径顺序完成一个节点的全部属性后，才开始下一个节点。本层已生成节点只经 shifted query 读取，到下一层才进入可读 context。每个节点都会重新计算主干和该层 query 前缀，每个属性也重算属性 decoder 前缀；当前未实现 K/V cache。

输出的 `rounds` 仅为层数，`node_steps` 为节点主干调用次数，`attribute_steps` 为实际属性决策次数。OBJECT 为 9 次属性决策、SPLIT 为 7 次、EMPTY 为 1 次。普通 N≥2 的树需生成 `2N-2` 个非 Room 节点，属性决策数为 `9N + 7(N-2)`。当前采样已包含层内串行，不能再以 O(D) 次串行决策描述，更不能宣称 O(log N) 推理耗时。

## 数据和旧检查点

构树、量化与编码格式未改，已经修正的 `datasets/processed_reltree_bedroom_v2` 可直接复用，无需重新预处理。新架构需要从头训练，当前全量配置输出为 `runs/bedroom_ar_8x512_cd10`。checkpoint 保存架构、cond_drop 和属性 decoder 配置，恢复时核对，旧架构权重不能作为本架构的断点继续训练。

`load_model()` 检测没有 `architecture` 标记的旧 checkpoint，使用冻结在 `legacy/` 中的旧并行模型和采样代码，供加载与推理对比。`runs/bedroom_full_v2` 的旧模型结果保留；旧版 teacher forcing 需配套 `legacy.packing` 和 `legacy.loss`。

## 原始数据接入

源头修复在根目录 `preprocess.py`、`utils.get_room_attributes()` / `embed_obj_token()` 和 `preprocess_script/geometry.py`。旧 `processed_*_wo_lat` 中混用了资产原点、世界 AABB 尺寸与 yaw，必须重新生成。

1. 上游从 `datasets/Levels2021` 读取原始几何，旧 processed 目录可提供场景名单与 split。
2. 由资产局部尺寸、中心和实例 `T @ R @ S` 得到 OBB 中心、局部完整边长、yaw；不再要求旋转后包围盒的 AABB 与 mesh AABB 相等。
3. 优先使用元数据中的资产中心。缺失时只用训练集轴对齐实例恢复中心，保存来源和一致性检查结果；不能可靠恢复的场景写入失败清单。细节见 [上游预处理](../../preprocess_script/README.md)。
4. 上游 NPZ 保存 `asset_local_obb_trs_v2` schema、类别顺序、房间边界和 frame。room 水平坐标框只由边界决定；高度使用声明的 3 m、地面 y=0 条件，不读取可能由家具决定的原始 room bbox 高度。
5. RelTree `prepare.py` 直接读取修正后的 NPZ，构树、量化。资产 ID 和 latent 不进入 token/query。上游同时修正电视柜与多人沙发的类别映射。

预处理保存 `preparation_report.json`：尝试/成功/失败数量、逐场景失败原因、裁剪率、量化重建失败、构树设置，以及训练/验证来自同一房屋的情况。现有 split 是房间级划分；正式泛化评估需另定房屋级划分，当前短训仅用于验证训练链路。

默认使用固定量化范围。若传 `--calibrate`，只在 train 场景拟合范围，并保存共享 `tokenizer.json`；val/test 直接使用该配置。

## 使用 Slurm 运行

本环境的登录进程不能直接使用 CUDA。通过 `sbatch` 申请 `general` 分区、`advanced` QoS，批处理脚本再执行 `.sh` 中的 Python 命令。默认 Python 为 `/home/cscgbnu/.conda/envs/arscene/bin/python`，可用 `RELTREE_PYTHON` 覆盖。

从仓库根目录操作：

```bash
mkdir -p networks/reltree/slurm

# 2 张 GPU，验证分配、CUDA 前向与反向
sbatch networks/reltree/reltree_gpu_check.slurm

# 1 张 GPU，完整机制测试（包含实际中断恢复）
sbatch networks/reltree/verify_resume.slurm

# 1 张 GPU，用已有 v2 数据验证新架构短训、过拟合后串行复现、断点恢复
sbatch networks/reltree/validate_ar.slurm

# 上游 NPZ 重建（CPU；保留原有 split）
sbatch preprocess_script/preprocess_levels.slurm --filter_fn bedroom \
  --split-root datasets/processed_bedroom_wo_lat --asset-center-tolerance .003 \
  --output datasets/processed_bedroom_wo_lat_v2

# 上游完成后，构树与量化（CPU）
sbatch networks/reltree/prepare_reltree.slurm --output datasets/processed_reltree_bedroom_v2

# 待预处理完成后，提交默认模型的单 GPU 训练
sbatch networks/reltree/train_reltree.slurm --config training_configs/reltree_config.yaml
```

已经准备好的数据可指定路径和输出目录：

```bash
sbatch networks/reltree/train_reltree.slurm \
  --config training_configs/reltree_config.yaml \
  --data-root datasets/processed_reltree_bedroom_smoke_v2 \
  --output networks/reltree/runs/my-run --steps 200

sbatch networks/reltree/train_reltree.slurm \
  --config training_configs/reltree_config.yaml \
  --data-root datasets/processed_reltree_bedroom_smoke_v2 \
  --output networks/reltree/runs/my-run \
  --resume networks/reltree/runs/my-run/last.pt --steps 400
```

当前为单 GPU 训练，不是 DDP。GPU 不可用时明确报错，不自动切换 CPU。输出含 `best.pt`、`last.pt`、共享 tokenizer、运行配置、逐步指标、总结与生成样本。检查点恢复模型、optimizer、scaler、步数、epoch、已消费样本数和 RNG；每个 epoch 使用 seed + epoch 的确定性排列，从第一个未消费的样本继续。旧 checkpoint 的样本位置由原 batch size 和 batch 计数恢复，包含最后一个不满 batch 的边界。

默认仍要求 batch size 一致。显式设置 `training.allow_batch_size_change: true` 时允许调整 batch size，保留模型、optimizer、学习率、RNG 和确切样本位置；训练样本、seed、tokenizer、train/val 内容指纹及 geometry schema 仍必须一致。变更前的 checkpoint 单独备份为 `before_batch_<old>_to_<new>_step_<step>.pt`。按 epoch 训练时，步数上限重新按剩余样本与剩余 epoch 计算，总目标不变；因 batch 分组变化，后续优化轨迹会改变。

## 全量训练与 tmux

`training_configs/reltree_full.yaml` 读取全部有效 train/val，8 层 512 维、cond_drop 0.1，单 GPU BF16，当前 batch size **128**，学习率 0.0003，目标为累计 **100,000 optimizer steps**（包含恢复前已完成的步数）。`max_steps: 100000`，`epochs: 1000` 提供足够高的 epoch 上限，避免原 200 epochs 上限提前结束训练。每 100 步及 epoch 结束保存 `last.pt`，每 1000 步跑全量验证并更新 `best.pt`。检查点先写临时文件，再原子替换正式文件；随机条件丢弃的 RNG 状态也随检查点恢复。

```bash
tmux new-session -s reltree-full
bash networks/reltree/run_full_pipeline.sh
```

流水线通过 `sbatch --wait` 顺序执行 GPU 测试、上游全量重建、RelTree 编码、训练。已有 v2 数据且存在 `READY` 标记时复用该数据，跳过对应预处理。两阶段预处理分别使用 CPU、时限 2 小时；训练每次申请 1 张 GPU，时限 **48 小时**。Slurm 在到时限前 180 秒给训练进程发送 USR1；训练在安全的 optimizer step 边界保存并以 75 退出，流水线自动重新提交并恢复 `last.pt`。硬超时或抢占也可从最近已保存的检查点重提；普通程序错误或人工取消不会无限重试。

流水线记录 `runs/bedroom_ar_8x512_cd10/pipeline.log` 和各阶段 `*_job.txt`。关闭终端、从 tmux detach 不影响任务；使用对应 tmux 会话查看流水线。训练日志位于 `slurm/reltree-full-<jobid>.out`。已有 v2 数据时直接启动全量训练，或手动恢复，只需：

```bash
sbatch networks/reltree/train_full_reltree.slurm
```

这个入口自动检查 `runs/bedroom_ar_8x512_cd10/last.pt`，有则恢复、无则从头训练，并使用文件锁防止两个训练进程同时写同一目录。正式训练的数据检查结果仍应以 `preparation_report.json` 为准；全量转换会记录几何不兼容的候选场景，不代表所有候选都被接纳。

### 12 层 768 维后置训练

`training_configs/reltree_12x768.yaml` 配置主干 12 层、768 维、12 个 attention heads，FFN 为 768 → 3072 → 768；属性 decoder 为 2 层、256 维、4 个 heads，FFN 为 256 → 1024 → 256。总参数 **91,574,817**，其中主干 Transformer blocks 为 85,054,464，属性 decoder 为 2,651,937。使用同一 v2 数据集，batch 128、BF16、cond_drop 0.1、学习率 0.0003，总目标 100,000 steps，epoch 上限 1000。新模型从头训练，后续只从独立运行目录 `runs/bedroom_ar_12x768_attr256_cd10/last.pt` 恢复。

```bash
# 在持久 tmux 会话中运行；16261 是本次 8 层训练的前置作业。
tmux new-session -s reltree-ar-12x768-cd10
bash networks/reltree/run_large_pipeline.sh 16261
```

该脚本通过 `sbatch --dependency=afterok:16261 --kill-on-invalid-dep=yes` 提交 `train_large_reltree.slurm`，不会在前置作业仍运行时训练新模型。第一次分配 GPU 后的超时/抢占恢复不再重复绑定旧依赖。前置作业失败会使这次依赖失效，不会误当作训练完成而启动大模型。每次申请 48 小时；模型/optimizer/RNG 的保存恢复和管道重提机制与 8 层版本一致。

2026-09-26 23:19（Asia/Shanghai）已提交作业 **16288**，Slurm 核实状态为 `PENDING (Dependency)`，依赖 `afterok:16261`。tmux 会话为 `reltree-ar-12x768-cd10`。配置与 tensor shape 已通过 meta tensor 检查，尚未启动该尺寸的 GPU 训练；记录见该运行目录的 `config_validation.json`。当前作业 ID 见 `train_job.txt`，前置 ID 见 `after_job.txt`，流水线日志见 `pipeline.log`；训练日志为 `slurm/reltree-12x768-<jobid>.out`。

历史 v1：2026-09-25，转换作业 16208 接纳 train 12,042 / val 1,760 个场景，使用了后来发现过严的 AABB 相等检查。旧训练作业 16211 经用户授权在第 5,072 步保存并退出，旧数据和检查点留存，不作为 v2 的续训起点。GPU 作业 16213 的 56 项测试全部通过，包含真实资产 8988、非对称几何、负缩放、原始场景不变、上游 NPZ 到 RelTree 的完整链路，以及含 dropout 的中断恢复。

v2 数据与旧并行基线：上游作业 16214 生成 train 12,999 / val 1,896 / test 3,724 个正确格式场景，8988 的 30 个原误筛场景全部恢复。作业 16215 将全部有效 train/val 转为 RelTree，构树失败、量化解码失败均为 0。旧并行模型作业 16216 已完成 162,600 步（200 epochs），输出 `runs/bedroom_full_v2`；这不是当前自回归架构的训练结果。数据缺失中心等上游失败项见独立报告，不计入有效训练样本。

## 2026-09-26 自回归架构验证

全部通过 Slurm 在 NVIDIA A800-SXM4-80GB 上运行，两个作业均 `COMPLETED`、退出码 `0:0`。

| 作业 | 验证内容 | 结果 |
| --- | --- | --- |
| 16252 | 67 项测试 | 全部通过，包含平台＋阶梯 mask、多层间接泄漏、shift 每层重置、属性因果性、padding 和打包顺序不变性 |
| 16252 | 一次训练 forward 与逐节点、逐属性前缀解码 | eval 模式 logits 在数值容差内一致；前序节点/属性确实影响后续预测 |
| 16252 | 中断恢复与旧 checkpoint | 含 dropout 的真实 USR1 中断恢复与连续训练一致；旧模型加载保持原有 logits |
| 16253 | 默认 624 万参数模型，64 train / 16 val，BF16，80 步 | train CE 40.8420 → 21.1691；val CE 40.6469 → 26.2980；4 个验证条件均完成树展开 |
| 16253 | 小模型单场景过拟合，400 步 | train CE 39.4656 → 0.001224；验证 CE 上升，不用于评价泛化 |
| 16253 | 从第 400 步恢复到 405 步，再自由串行生成 | 完整复现该训练场景的类型、类别、7 维 bin 与树拓扑：6 个物体、5 层、10 次节点主干调用、82 次属性决策 |

这些结果验证实现链路和训练/采样一致性；在该次验证结束时尚未启动新架构的全量训练。单场景复现的是量化后的目标 token，仍有 tokenizer 的量化重建误差。

完整记录：[机制测试日志](slurm/reltree-resume-test-16252.out)、[GPU 集成验证日志](slurm/reltree-ar-check-16253.out)、[指标汇总](runs/validation-ar-16253/validation_summary.json)、[串行复现场景](runs/validation-ar-16253/overfit_generation.json)。复现入口为 `verify_resume.slurm` 和 `validate_ar.slurm`，Python 命令保存在对应 `.sh` 中。

## 2026-09-26：8×512、cond_drop 10% 全量训练

启动前的 GPU 作业 16255 通过全部 73 项测试，包含场景级条件丢弃、条件信息隔离、eval 保留条件、旧 AR checkpoint 加载，以及含条件丢弃和网络 dropout 的中断恢复一致性。作业 16256 用全量模型配置短训 80 步（64 train / 16 val）：train CE 40.9688 → 20.0745，val CE 41.3092 → 26.5220；小模型单场景过拟合、恢复和 token 精确复现也通过。两个验证作业均退出码 `0:0`，记录见 [验证汇总](runs/validation-ar-16256/validation_summary.json)。

全量训练作业 **16257** 于 2026-09-26 13:49（Asia/Shanghai）启动，使用单张 A800。tmux 会话为 `reltree-ar-8x512-cd10`，复用现有 v2 数据，输出 `runs/bedroom_ar_8x512_cd10`。配置为 8 层 512 维主干、2 层 128 维属性 decoder、10% 条件丢弃、batch size 16、BF16、200 epochs。每次 Slurm 申请 48 小时，流水线支持自动断点续训。

```bash
tmux attach -t reltree-ar-8x512-cd10
tail -f networks/reltree/slurm/reltree-full-16257.out
```

启动后进度以 [训练指标](runs/bedroom_ar_8x512_cd10/metrics.jsonl) 和 [流水线日志](runs/bedroom_ar_8x512_cd10/pipeline.log) 为准；重新提交后的作业编号记录在 `runs/bedroom_ar_8x512_cd10/train_job.txt`。

2026-09-26 13:56 按用户要求将 batch size 从 16 调为 **128**。75 项 GPU 测试（作业 16258）通过后，用 USR1 请求作业 16257 在第 **4000** 步保存退出；这是配置切换，未触及 48 小时时限。流水线自动启动作业 **16259**，恢复同一模型、optimizer、学习率及 RNG，从 epoch 4（从 0 计数）的第 11,968 个已消费样本之后继续。切换前检查点保留为 `before_batch_16_to_128_step_4000.pt`。当时总目标为 200 epochs，步数上限为 **23,899**，包括之前 batch size 16 的 4,000 步；该阶段日志为 `slurm/reltree-full-16259.out`。

2026-09-26 14:57 按用户要求将目标延长至累计 **100,000 steps**，设置 `max_steps: 100000` 和 `epochs: 1000`。原配置保存为运行目录中的 `config_before_100k.yaml`。作业 16259 收到 USR1 后在 **14,134** 步保存退出；流水线自动提交作业 **16261**，从同一断点的 epoch 104、已消费 3456 个场景处继续。启动日志确认目标为 100,000，且已继续更新到 14,200 步。主干、条件丢弃、batch size、优化器和学习率保持原配置，日志为 `slurm/reltree-full-16261.out`。此次退出用于加载新的训练预算；旧 tmux shell 的“time limit/preemption”提示是旧版通用文案，未触及时限。

## 2026-09-24 旧并行架构实测（历史记录）

所有计算通过 Slurm 在 A800-SXM4-80GB 上完成：

| 作业 | 内容 | 结果 |
| --- | --- | --- |
| 16205 | 两张 GPU 的 CUDA 前向/反向 | 两张均通过；PyTorch 2.5.1，CUDA build 11.8 |
| 16206 | 47 项测试 | 全部通过，包含多层泄漏、同层目标扰动、padding、并行/逐尺度 logits、动态展开、根联合约束和 checkpoint |
| 16206 | 24 个真实 train / 7 个 val，64 维 2 层，120 步 | train loss 41.2181 → 7.8265；val 40.8655 → 35.2482；第 40 步 val 为 29.9877，随后出现过拟合 |
| 16206 | 一个真实场景过拟合，200 步 | train loss 41.2960 → 0.006796；此项验证可学习性，不衡量泛化 |
| 16206 | 从第 200 步恢复到 205 步 | train loss 0.006796 → 0.006570，optimizer/权重恢复成功 |
| 16207 | 默认 256 维 6 层 BF16，20 步 | train loss 41.4906 → 26.7843；val 41.3347 → 30.0907 |

各短训结束后，对 4 个验证条件进行动态生成均完成结构展开；尚未评估生成分布与物理质量。40 个预处理候选中有 9 个因原始 bbox 与 yaw-only OBB 不一致而被拒绝；成功的 31 个场景无量化裁剪或解码失败。失败清单和共享房屋信息保存在预处理报告中，不能把本次小样本结果作为论文实验。

日志：`slurm/reltree-gpu-check-16205.out`、`slurm/reltree-smoke-16206.out`、`slurm/reltree-train-16207.out`。checkpoint 和 JSON 指标：`runs/smoke-16206/{train,overfit,resume}/`、`runs/default-validation/`。这些生成产物被本目录 `.gitignore` 排除。
