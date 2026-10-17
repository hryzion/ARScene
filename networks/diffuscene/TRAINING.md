# Livingroom 无条件训练

2026-09-29的新训练采用900k步、120k步LR减半和旋转/排列增强，详见
[TRAINING_900K.md](TRAINING_900K.md)。下文保留原100k运行的说明。

配置：`training_configs/diffuscene_livingroom_uncond_100k.yaml`。
`train_diffuscene.py` 检测到 `training.max_steps` 后使用 `step_training.py`；
旧的按 epoch 训练入口仍可使用。

## 数据

本次从 Levels2021 重新执行 OBB 预处理，`--atiss` 增加两个类别保留槽：

```bash
sbatch preprocess_script/preprocess_levels.slurm \
  --atiss --get_stats --filter_fn livingroom \
  --raw-root datasets/Levels2021 \
  --split-root datasets/processed_livingroom_wo_lat \
  --asset-center-tolerance .003 \
  --output datasets/atiss_livingroom_obb_v2_wo_lat
```

输出目录必须为空；已有 `READY` 时直接使用完成的数据。Slurm 16343 已完成：
train 5,642、val 810、test 1,633；1,276 个候选房间失败，详见数据目录内的
`preparation_report.json`。旧目录只提供场景名单与划分，不读取旧几何。

每个物体保存 33 维类别、3 维 OBB 中心、3 维完整边长、1 维弧度 yaw。
本次训练只加载布局；位置与尺寸范围仅在训练集拟合，yaw 直接计算 cos/sin。
模型输入为 41 维。容量取 train/val 最大物体数，并在
`run_summary.json`、`data_audit.json` 和运行目录的解析后 `config.yaml` 中记录；
不读取 test，不截断物体。未占用槽使用最后一个类别表示，保留原 DiffuScene
padding 协议。

## 网络与训练

- U-Net `dim=512`，`dim_mults=[1,1,1,1]`，batch size 128，单卡 BF16。
- 100,000 次 optimizer update；1000 个扩散时间步，v prediction，属性分项 MSE。
- Adam，初始学习率 2e-4，每 10k optimizer step 乘 0.5，梯度范数裁剪 10。
- 房间边界、文本、局部物体、布局安排条件全部关闭；不建立 ResNet/BERT。
  固定槽位的 128 维可学习 embedding 不包含场景条件。
- 本基线关闭旧的辅助 AABB IoU loss：旧实现按半尺寸处理盒子，而当前数据
  保存完整 OBB 尺寸。此次不改变该辅助损失的数学定义。
- 每 20 step 写训练指标，每 1000 step 评估完整 val，每 500 step 保存
  `last.pt`；第一个 step 也保存。按验证 loss 更新 `best.pt`。
- 验证固定扩散噪声种子，并恢复训练 RNG，使验证不改变后续训练噪声序列。

## 启动与恢复

所有 Python 命令在 `.sh` 中，通过 Slurm 分配的 GPU 运行。
额外依赖 `einops==0.8.1` 安装在项目 `.runtime/diffuscene`，启动脚本自动设置
`PYTHONPATH`。如需重建该独立依赖目录：

```bash
bash networks/diffuscene/install_runtime.sh
```

数据完成后先验证：

```bash
sbatch networks/diffuscene/verify_livingroom.slurm
```

验证涵盖连续训练与恢复训练的模型/optimizer 精确一致、外部条件变化不影响
无条件 loss、短扩散采样，以及实际 512 宽度、batch 128、全量 train/val 的
训练与评估。成功后写入运行目录的 `VERIFIED`。

在 tmux 中运行持久监控脚本：

```bash
tmux new-session -d -s diffuscene-livingroom \
  -c /home/cscgbnu/zhx/code/ARScene \
  'bash networks/diffuscene/run_livingroom_pipeline.sh'
```

每次申请独立的一张 GPU，单次 Slurm 时限 48 小时。`last.pt` 包含模型、Adam
状态、step、epoch 内样本位置、随机数状态、normalizer、数据指纹及配置。
重启相同命令会从 `last.pt` 恢复；到 100k step 停止。BF16 不使用梯度缩放。
收到 USR1/TERM 时在 optimizer update 边界保存，返回 75；监控脚本对该退出码、
TIMEOUT、PREEMPTED 自动重新申请资源。普通错误或手动取消不会自动重启。

运行目录：`networks/diffuscene/runs/livingroom_uncond_100k/`。
日志目录：`networks/diffuscene/slurm/`。

```bash
tmux attach -t diffuscene-livingroom
tail -f networks/diffuscene/runs/livingroom_uncond_100k/metrics.jsonl
```

新 `.pt` 是完整训练 checkpoint，模型权重位于 `checkpoint['model']`；不能直接
作为旧采样入口期望的裸 `state_dict` 加载。采样应采用 checkpoint 中的解析后
network 配置和同一 normalizer。
