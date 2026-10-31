# SceneFoundry baseline on ARScene

训练入口为 `train_scenefoundry.py`，配置在 `training_configs/`，Slurm 脚本在
`training_scripts/`，权重和日志写入 `pretrained/scenefoundry/`，与项目现有
ATISS / MiDiffusion baseline 使用同一套组织方式。

## 来源与复现范围

网络源自本地 `../../public_code/SceneFoundry`，提交
`2dfd31ac7de1ef8236b87703675e9f609ad481c5`。移植了其 `models/networks/` 下的
U-Net、DDPM、场景训练封装、IoU 函数和房间特征提取器，不依赖原仓库的绝对路径。
采样时的碰撞 guidance 则直接加载上述原仓库的 `CollisionOptimizer`，需要保留该
checkout，并编译它依赖的原版旋转 IoU CUDA 扩展；详见 [GUIDANCE.md](GUIDANCE.md)。
该公开提交没有完整场景训练启动脚本，因此这里使用 ARScene 的数据和训练管理流程。

保留的训练行为：

- 原 U-Net1D，默认 dim=512、四级 `[1,1,1,1]`，实例 embedding=128。
- **eps prediction**，linear beta=0.0001→0.02，1000 个扩散时间步。
- 分项 MSE：bbox + class；开启原有效物体对 IoU 正则，系数 **0.5**，
  保留 alpha_bar(t) 权重及原代码不裁剪 x_recon 的行为。
- 原代码 batch=1 时将布局/条件重复32次后采样不同噪声的行为。
- 房间条件配置使用原 ResNet18 结构、固定 BN 统计、随机初始化权重。

本次是 **ARScene 数据上的 layout-only baseline**。默认采用其他 baseline 的
livingroom 无条件任务，另附 `scenefoundry_livingroom_roomshape_eps_900k.yaml`
用于原实现的房间掩码条件任务。公开 SceneFoundry livingroom 配置使用房间条件、
25个类别通道及32维形状特征；这里使用项目的31类家具+START/END共33通道，不训练
缺失的形状 latent。900k 次更新、每120k次更新学习率减半与项目 DiffuScene 900k
实验对齐，不将上游配置的 epoch 数误当成 optimizer update 数。

训练不使用推理 guidance。完整的交互对象资产检索、约束优化和世界扩展流程不属于
本训练 baseline；文本、补全、重排训练配置会明确拒绝。上游文本注意力已被移除，
不能通过打开文本配置复现一个不存在的训练分支。

## 数据协议

- 数据根目录：`datasets/atiss_livingroom_obb_v2_wo_lat`，要求 `READY` 及
  `asset_local_obb_trs_v2` schema。
- 使用 `train` 训练、`val` 验证，训练和统计不读取 `test`。
- 类别从项目 one-hot 转为 **-1/+1**；最后一个 END 通道编码空槽位。
  保留项目 START 通道，真实物体不允许使用 START/END。
- 位置/完整边长按真实训练物体拟合 min/max；yaw 保持弧度再计算 `[cos, sin]`。
  空槽位的位置、尺寸和两个角度通道均为0。
- **仅在 IoU 计算时将完整边长除以2**，使其与上游半尺寸几何一致；归一化统计及
生成结果仍采用项目完整边长。IoU 仍为上游 axis-aligned 正则，不是旋转 OBB IoU。
- 使用项目的四方向旋转和物体排列增强，统计覆盖旋转后的训练位置范围；房间掩码
  同步旋转。随机性由 seed/epoch/index 决定，不受 worker 预取影响。
- 容量从 train/val 最大物体数确定，当前为27；不截断场景。
- 无条件数据不打开房间掩码；有条件数据使用64×64单通道、范围 `[0,1]` 的掩码。

统计同时嵌入 checkpoint 与保存后的 `config.yaml`，IoU 和推理不需要原作者的
`dataset_stats.txt`。推理必须加载同一个 checkpoint 的 normalizer，不能另拟合测试集。

## Slurm 启动

在仓库根目录执行。默认环境是 `/home/cscgbnu/.conda/envs/arscene/bin/python`，
通过已有 `.runtime/diffuscene` 提供 einops；可用 `SCENEFOUNDRY_PYTHON` 指定其他
包含 torch、torchvision、numpy、Pillow、PyYAML、tqdm、einops 的环境。
如果当前项目尚未安装 einops，可运行现有 `bash networks/diffuscene/install_runtime.sh`。

先验证数据、原代码数值一致性、续训、采样及真实完整模型：

```bash
mkdir -p pretrained/scenefoundry/slurm
sbatch training_scripts/scenefoundry_verify.sbatch
```

单次正式训练分配一张 GPU，FP32、batch=128、Adam lr=2e-4、gradient clip=10：

```bash
sbatch training_scripts/scenefoundry_livingroom_eps_900k.sbatch
```

需要跨48小时分配自动续训时，在 tmux 中运行监督脚本：

```bash
tmux new-session -d -s scenefoundry-eps-900k \
  'bash training_scripts/run_scenefoundry_livingroom_eps_900k.sh'
```

房间条件实验使用同一个启动器，输出到独立目录：

```bash
SCENEFOUNDRY_CONFIG=training_configs/scenefoundry_livingroom_roomshape_eps_900k.yaml \
 bash training_scripts/run_scenefoundry_livingroom_eps_900k.sh
```

与 ATISS / DiffuScene / MiDiffusion 的 room-condition 对比使用独立的 500k 配置：
`training_configs/scenefoundry_livingroom_roomshape_eps_500k.yaml`。
保留 eps prediction、FP32、batch 128、旋转/排列增强和 SceneFoundry IoU 损失；
输出到 `pretrained/scenefoundry/livingroom_roomshape_eps_500k/`。

```bash
tmux new-session -d -s scenefoundry-room-eps-500k \
  'bash training_scripts/run_scenefoundry_room_condition_500k.sh'
```

监督脚本通过 `training_scripts/scenefoundry_room_condition_500k.sbatch` 分配一张 GPU，
在 Slurm 超时或抢占后从 `last.pt` 续训，达到 500,000 次更新后结束。

训练入口要求 Slurm 分配的 CUDA GPU，不会在登录节点回退到 CPU。
重复执行默认从 `last.pt` 恢复；改变网络、目标、数据、归一化、batch或学习率协议
会被拒绝，避免把不同实验混成一个。Slurm提前180秒发送USR1时保存进度，退出码75
由监督脚本处理；超时或抢占也可以续训，其他失败会停止并保留日志。

输出目录默认是 `pretrained/scenefoundry/livingroom_unconditional_eps_900k/`：

| 文件 | 内容 |
| --- | --- |
| `last.pt` / `best.pt` | 模型、Adam、step、下一条数据位置、随机状态、配置、统计 |
| `step_*.pt` | 每100k次更新保留的 checkpoint |
| `config.yaml` / `data_audit.json` | 实际配置、数据指纹、容量和编码协议 |
| `normalizer_stats_atiss.json` | 当前模型对应的训练统计 |
| `metrics.jsonl` | 分项训练/验证损失、梯度范数、lr、速度和显存 |
| `run_summary.json` | GPU、Slurm任务号、参数量和数据规模 |
| `VERIFIED` | GPU集成验证报告路径 |
| `COMPLETE` | 已完成配置中的总更新预算 |

## 加载 checkpoint

### Room-condition 1000 场景评测

复用 ATISS / DiffuScene / MiDiffusion 的同一份测试房间 `conditions.json`，使用
`livingroom_roomshape_eps_500k/best.pt`、FP32、batch=128、1000 步 DDPM。
房间 mask 按训练时的 RGB 双线性缩放、第一通道、除以 255 处理，实际输入为
`[B,1,64,64]`。碰撞和 room-layout guidance 直接调用原仓库 `CollisionOptimizer`，
保留 batch mean，均仅最后 10 步启用；权重分别为源码默认的 2 和 300000，
不指定物体数量。房间边界转换为顺时针 XZ 顶点，直接调用原仓库
`Room.calc_box_from_polygon(points, 50)`，按源码补零到 `[B,50,7]`。
不修正原版包围盒预处理的数值误差，也不对输入边界做吸附、简化或其他修补。

```bash
sbatch sampling/scenefoundry_room_condition_1000.sbatch \
  pretrained/room_condition_500k/evaluation_best_1000_20261012_232434
```

结果保存在评测根目录的 `scenefoundry_collision_room_guidance/`，原来的
`scenefoundry/` 碰撞 guidance 结果保留。脚本第二参数可指定输出目录，第三参数可选
`collision_room`（默认）、`collision` 或 `none`。作业依次执行采样、项目 OBB collision /
footprint OOB 评测、独立解码核对；分别输出 `report.json`、
`collision_oob_metrics.json` 和 `decode_audit.json`。`guidance_verification.json`
核对原版类/方法来源、原版 CUDA IoU、room-layout 梯度及 batch mean；
`room_guidance_audit.json` 记录外部包围盒来源、覆盖误差和哈希（只记录，不修正）。
原始预测、实际 mask 和外部包围盒保存在 `raw/batch_*.npz`。
有效槽位沿用 SceneFoundry 的 signed END < 0 判定。

```python
import torch
from networks.scenefoundry.step_training import load_for_inference

model, normalizer, config = load_for_inference(
    'pretrained/scenefoundry/livingroom_unconditional_eps_900k/best.pt', 'cuda')
net = config['network']
with torch.no_grad():
    samples = model.sample(
        room_mask=torch.zeros(2, 1, 64, 64, device='cuda'),
        num_points=net['sample_num_points'], point_dim=net['point_dim'], batch_size=2)
```

`samples` 为 `[B,N,41]`，依次是 translation3、size3、cos/sin2、class33。
无条件模型忽略示例掩码；房间条件模型需要相同编码的真实掩码。END分数>=0的槽位
为空；解码真实类别时只在31个家具类别中取argmax，yaw=`atan2(sin,cos)`，几何使用
所加载 normalizer 还原。不要将它送进项目针对0/1编码的旧 DiffuScene 解码流程。

带 guidance 的 Slurm 批量采样入口为 `sampling/scenefoundry_sample_guidance.sbatch`，
直接调用原版 `CollisionOptimizer`，保留 batch 平均损失和原始梯度计算。
默认最后10步做旋转框碰撞 guidance，可选最后100步做物体数量控制；由于原版
数量损失只访问第一个场景，数量控制要求采样 `--batch-size 1`。
具体调度、与论文完整流程的差异及几何适配见 [GUIDANCE.md](GUIDANCE.md)。

## 验证与兼容修复

`tests/test_scenefoundry_training.py` 在 Slurm GPU 上比较原仓库与移植 U-Net 的输出和
参数梯度，以及完整尺寸适配后的 eps/IoU 损失和梯度。数据测试覆盖类别、空槽位、
真实yaw、训练统计隔离、增强确定性和无条件掩码隔离。
`verify_training.py` 使用真实数据检查中途/epoch边界恢复，包括改变worker数量；
然后验证checkpoint加载、DDPM采样、房间条件训练，以及完整75,557,289参数模型的
batch=128训练与全验证集计算。

除数据接口外，上游文件的兼容修复包括：移除训练不需要的GUI/CLIP/碰撞资产导入；
修正无guidance时访问None的采样错误；修正cosine schedule遗漏赋值；去掉普通采样
中未使用的整条轨迹缓存；允许从配置读取统计；feature extractor工厂只创建选中的
模型并使用当前torchvision参数名称。默认linear eps训练的U-Net数学保持不变。
