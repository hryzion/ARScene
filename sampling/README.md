# 两种模型的无条件批量采样

入口 `sample_layouts.py`，Python 命令位于 `run_sample_layouts.sh`，计算通过
`sbatch sample_layouts.slurm` 分配 GPU。默认各采样 1000 次，保留所有结果、
空布局和失败状态，不按质量筛选或补采替换。

## 2026-10-01：RelTree livingroom 跨场景批量采样

已使用客厅模型 `livingroom_root_ar_12x768_attr256_cd10/last.pt`（100k step）
串行生成 1000 个场景，结果保存在该运行目录的
`samples_last_uncond_1000_20261001/`。1000 个场景全部完成，11774 个物体；
房间尺寸 token、树 token 重新解码后与导出的布局一致。原结果保留。

采样入口现在支持多个场景共同前向计算，RelTree 默认 batch size 为 **64**，
DiffuScene 仍为 128；显式 `--batch-size 1` 使用原 RelTree 串行路径。
例如以后生成一批新结果：

```bash
sbatch --job-name=sample-reltree-living-batch sampling/sample_layouts.slurm \
  --model reltree --room-type livingroom \
  --checkpoint networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/last.pt \
  --output networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/samples_last_batch64_1000 \
  --count 1000 --batch-size 64 --seed 20260929
```

`--room-type` 是报告中的训练分布标签，不作为生成条件；模型由 checkpoint 决定。
`RootARRelTree.generate_batch()` 将每个活跃场景的下一个节点一起送入网络，
每个场景内部继续按原顺序生成节点及属性。历史、Room、父节点索引、随机数流、
停止状态彼此独立；已结束/失败的场景退出当前 batch，返回结果保持输入顺序。
该实现适用于 `room_root_ar_v3`，不需要重新训练。

每个场景的种子仍为 `seed + index`。相同 batch 配置下可复现并支持恢复；
未完成的批次按原成员及种子整体重做，末尾仅生成剩余数量。
不同 batch size 的浮点计算路径可能略有差异，BF16 下相同种子不保证逐 token
与串行结果一致。不要向同一输出目录混写不同 batch size 的结果，manifest 会拒绝。

GPU 验证作业 16544 已通过 18 项测试，覆盖串行对照、场景顺序/随机数隔离、
不同长度、生成上限、异常场景隔离、条件输入、padding 和自回归约束。
真实 last.pt 的 batch=1 BF16 结果与原实现完全一致；不同长度的因果历史在
FP32 单独/批量前向中的 hidden 最大误差为 2.38e-6；同一 batch 重复采样、
最后不足整批以及命令行部分批次恢复均已通过检查。

A800 80GB、BF16、每组 64 个场景，计时包含生成过程，排除模型加载、预热和导出：

| Batch size | 场景/秒 | 相对串行 | PyTorch 峰值 allocated 显存（MiB） |
| --- | ---: | ---: | ---: |
| 1 | 2.23 | 1.00× | 561 |
| 8 | 7.69 | 3.44× | 624 |
| 16 | 11.68 | 5.23× | 676 |
| 32 | 13.36 | 5.98× | 799 |
| 64 | 15.63 | 6.99× | 1031 |

这是当前 checkpoint 的实测吞吐量，场景长度会影响速度；allocated 不包含
全部 CUDA 上下文/缓存开销。报告和各组样本位于运行目录
`batch_sampling_check_20261001/`。复核命令：

```bash
sbatch sampling/verify_reltree_batch.slurm \
  --checkpoint networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/last.pt \
  --output networks/reltree/runs/livingroom_root_ar_12x768_attr256_cd10/batch_sampling_check_new
```

## 2026-09-29 的历史采样

2026-09-29 使用两个已完成 100k step 训练的 `last.pt`：

```bash
sbatch --job-name=sample-diffuscene-1000 sampling/sample_layouts.slurm \
  --model diffuscene \
  --checkpoint networks/diffuscene/runs/livingroom_uncond_100k/last.pt \
  --output networks/diffuscene/runs/livingroom_uncond_100k/samples_last_uncond_1000_20260929 \
  --count 1000 --batch-size 128

sbatch --job-name=sample-reltree-1000 sampling/sample_layouts.slurm \
  --model reltree \
  --checkpoint networks/reltree/runs/bedroom_root_ar_12x768_attr256_cd10/last.pt \
  --output networks/reltree/runs/bedroom_root_ar_12x768_attr256_cd10/samples_last_uncond_1000_20260929 \
  --count 1000 --batch-size 1
```

DiffuScene 对应 **livingroom**，RelTree 对应 **bedroom**，不是在相同房间分布上
训练的两个模型。默认推理为 BF16，seed=20260929。

- DiffuScene：原始 1000 步 DDPM，`clip_denoised=False`，最后一批按剩余数量
  生成，恰好 1000 次。调用 `sample(room_mask=None)`；不加载测试集、房间图、
  边界或外部 room frame。位置和完整 OBB 尺寸使用 checkpoint 的训练集统计
  恢复；yaw 是 `atan2(sin, cos)`，不做角度反归一化。
- RelTree：`room_root_ar_v3`，`condition=None`，从 SOS 生成 room 长宽高再展开
  布局；temperature=1，top-k=0。使用模型深度上限 32、物体上限 64、节点上限
  255，触及上限时保留部分布局及失败原因。输出的 room 框不表示生成了不规则
  房间边界。

每个输出目录包含：

| 文件 | 内容 |
| --- | --- |
| `scenes/000000.json` … `000999.json` | 逐场景布局、状态、物体、种子与耗时；RelTree 还包含 room、树和 token |
| `samples.jsonl` | 全部 1000 条结果 |
| `raw/batch_*.npz` | DiffuScene 的原始 41 维预测，保留无效值以供排查 |
| `report.json` | checkpoint step、数量、状态统计、模型和采样设置、计时 |
| `sampling_times.csv` | 每个样本的状态、物体数、计时和批次索引 |
| `manifest.json` | checkpoint SHA256 与固定运行参数，恢复时检查一致性 |
| `index.html`、`preview/page_*.png` | 按生成顺序的前 64 个样本预览，无筛选 |
| `COMPLETE` | 汇总、数量检查和预览全部完成的标记 |

JSON 物体的 `position` 是 OBB 中心、`size` 是完整边长，米制、y 向上、yaw
为弧度。输出和图片均为布局盒子，不是资产 mesh。`complete` 表示生成流程
完成，不保证无碰撞。DiffuScene 预测的保留 start 类、非有限值和非正尺寸会
计为失败并保留原始记录，不隐式裁剪或纠正。

计时在生成调用前后执行 `torch.cuda.synchronize()`；不含模型加载、热身、
文件导出和画图。RelTree batch=1 时每条是单场景延迟；RelTree batch>1 和
DiffuScene 每条是批次耗时除以批量大小，不是单场景独立推理延迟。不同批量大小和房间分布下不能直接把
两者比值当作模型速度提升。

中断后重跑相同命令会检查 manifest，跳过完整结果；DiffuScene 未完成的批次
按同一种子重做。更换 checkpoint 或采样参数时使用新目录。

采样前 GPU 回归验证：

```bash
sbatch sampling/verify_sampling.slurm
```

验证包括省略 room mask 与原无条件路径的一致性、条件模型拒绝缺失 mask、
弧度 yaw 与完整尺寸的正确解码，以及无效预测仍被记录。
