检查日期：2026-09-29。结论：当前 livingroom 无条件训练的数据字段、物体槽位、
损失输入及断点恢复未发现错位；新旧默认训练方案并不完全等价。

比较对象是 `train_diffuscene.py` 的旧 epoch 分支、
`ThreeDFrontDatasetDiffuScene`，以及新 `step_training.py` / `LayoutDataset`。
历史角度协议另对照 Git HEAD `9b143a0` 的 `SceneTokenNormalizer`。

本次 GPU 审计 Slurm **16414** 全部通过，原始结果：
[training_alignment_audit.json](runs/livingroom_uncond_100k/training_alignment_audit.json)。
只新增审计脚本与报告；原训练代码、100k checkpoint 和采样结果没有修改。

| 检查 | 实际结果 |
| --- | --- |
| 全量数据对齐 | train 5,642 场景 / 74,122 物体；val 810 场景 / 10,581 物体；数据指纹与训练时一致 |
| 新旧 loader | 相同 normalizer、容量 27 下，translations / sizes / angles / class_labels 全部逐元素精确一致 |
| 独立构造目标 | 从原始物体行独立计算归一化、cos/sin、类别和 padding，全部精确一致 |
| 新旧单步优化 | 相同无条件配置、FP32、初始权重、batch 和 RNG；真实新训练循环与旧 train_on_batch 的 loss、更新后参数、Adam 状态精确一致 |
| BF16 恢复 | epoch 中途及边界恢复后，模型、Adam、step、epoch、sample_offset、Torch/CUDA RNG 与连续训练精确一致；覆盖不足一批及 LR 衰减 |

优化测试使用 dim=32、dim_mults=[1,1]、5 个真实训练场景、batch=3，避免重跑大模型。
全量数据测试使用实际运行保存的 normalizer。当前数据缺少旧 loader 必需的
`room_shape.npz`，因此给旧类注入占位房间元数据，实际调用其未修改的
`__getitem__` 和 collator；不代表旧文件入口或房间条件网络可直接运行。

输入协议已经核对：原始行是 `[类别33, 位置3, 完整尺寸3, yaw1]`；进入扩散模型时
为 `[位置3, 尺寸3, cos(yaw), sin(yaw), 类别33]`，共 41 维。
网络和损失都使用 `0:3 / 3:6 / 6:8 / 8:41`。31 是保留 START 类，32 是 END/空槽类。
空槽在归一化后追加，位置/尺寸为 0，角度为 `[1,0]`，END=1，其余类别为 0；
参与扩散损失，与旧行为一致。两条路径都保留物体行顺序，没有自回归式目标平移。

`get_loss`、U-Net 的特征拆分/输出拼接以及 `diffusion_ddpm.py` 的训练数学逻辑
没有被新训练器改变。`loss_separate=true` 的实际损失为
`alpha_bar[t] * (MSE(bbox的8维) + MSE(class的33维))`（v prediction、IoU 关闭）。
日志中的 trans/size/angle 是分项统计，object loss 已包含在 class loss 内，
不是额外重复相加。

不能视为完全等价的差异：

| 项目 | 旧路径 / 历史版本 | 新的 100k 运行 |
| --- | --- | --- |
| 房间条件 | 旧 YAML 未指定开关，默认 room_mask_condition=true，context_dim=64 | 显式关闭房间/文本等外部条件，context_dim=0；保留 128 维槽位 embedding |
| IoU 辅助损失 | 旧 YAML 开启，使用外部统计文件，按 center±size 构造 AABB（半尺寸假设） | 关闭；当前数据是完整 OBB 边长 |
| yaw 协议 | Git HEAD 历史 normalizer 先 min/max 到 [-1,1] 再计算 cos/sin | 直接用物理弧度；当前工作区的旧入口也已改用 from_config 的 radians 协议 |
| 新拟合统计量 | 没有 stats 文件时，旧 fit_atiss 的通用 collator 把非全零 END padding 当有效行，使尺寸 min 含 0 | 在未 padding 的真实训练物体上拟合；本次核对与全部真实物体 min/max 精确一致 |
| LR 更新时机 | 每个 epoch 开头按累计 iterations 更新 | 每次 optimizer update 前按 step 更新；跨衰减点时可相差不足一个 epoch |
| 精度 / 随机序列 | FP32；os.listdir 场景顺序、全局 shuffle RNG；验证消耗训练 RNG | BF16；排序场景、按 epoch 固定 shuffle；验证保存/恢复 RNG |
| 验证 / 保存 | 实际读取 test，每 epoch 验证；裸 state_dict | 读取 val，每 1000 step 验证；完整 model/optimizer/RNG/normalizer checkpoint |

旧 padding 污染已用真实旧数据方法复现：新统计 size min 为
`[0.0010000183, 0.0400000401, 0.0010030565]`，旧含 padding 的现场拟合可得到
`[0,0,0]`。这不说明预先保存的旧 stats 一定有问题，但说明两种拟合入口不等价。

历史 minmax-yaw 权重必须配套历史角度协议和 stats。不能直接用当前 radians
normalizer 解码历史权重，也不能将新 `.pt` 当成旧裸 state_dict 加载。
当前新训练和 `sample_layouts.py` 都使用 checkpoint 配置、匹配的 normalizer、
`atan2(sin, cos)` 和 END 类 argmax，协议一致。

已有的完整 100k checkpoint GPU 接口审计（Slurm 16413）另外验证了
训练/采样条件 `[B,27,128]`、41 维目标以及 v prediction 的正反变换：
[interface_audit.json](runs/livingroom_uncond_100k/samples_last_uncond_1000_20260929/interface_audit.json)。
其中旧 `delete_empty_from_network_samples` 的 END 符号判断和类别维数问题仍存在；
当前 `sample_layouts.py` 不调用该接口。

重跑本次检查：

```bash
sbatch networks/diffuscene/audit_training_alignment.slurm
```

入口：[audit_training_alignment.py](audit_training_alignment.py)。审计中生成的小模型
checkpoint 使用临时目录，结束后自动清理，结果 JSON 保存到实际运行目录。
