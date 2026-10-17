# 模型对应的角度处理

训练和采样入口使用 `SceneTokenNormalizer.from_config(config, ...)`，从同一份
YAML 的 `model.type` 或 `network.type` 选择角度协议，不需要手动注释源码。

| 模型 | 训练时的 yaw | 采样后的 yaw |
| --- | --- | --- |
| InteriorGPT / SAR、VQVAE、ATISS | min/max 映射到 `[-1, 1]` | 反归一化恢复弧度 |
| DiffuScene、FlowScene、MiDiffusion（含 mixed）、PhyScene | **跳过 rotation 归一化**，直接对原始弧度计算 `[cos(yaw), sin(yaw)]` | `atan2(sin, cos)` 直接得到弧度，**跳过 rotation 反归一化** |
| Drift | 当前标量 yaw 路径使用 min/max | 使用相同 min/max 恢复 |
| RCVAE | 原有 mean/std 路径保留弧度，并附加 sin/cos | 原有 `inverse_transform` 恢复 |
| RelTree | 独立的相对几何与量化 tokenizer | 不使用此 normalizer |

DiffuScene 系列的位置和尺寸仍按 min/max 归一化。`transform_atiss` 保留标量
弧度，数据集 collator 再计算 cos/sin；`invert_transform_atiss` 只恢复位置、
尺寸，保留 `atan2` 的结果。例如 `yaw = π/2` 编码为 `[0, 1]`，解码回 `π/2`。
圆周角往返按模 `2π` 等价，不能要求 `2π` 解码后仍是 `2π` 而不是 `0`。

`fit_atiss`、`transform_atiss`、`invert_transform_atiss` 共用同一个字段选择
规则。DiffuScene 系列不需要 rotation 的 min/max；即使旧统计 JSON 中包含
rotation，默认也不会使用它缩放角度。MiDiffusion 的辅助统计文件在此协议下
将 `bounds_angles` 写为 `[-π, π]`，不依赖不存在的 rotation 统计项。

统计 JSON 新增 `__normalizer__` 元数据，保存模型标识、类别维度和角度协议；
读取带元数据的文件时检查协议是否匹配。旧的无元数据 JSON 仍可读取，角度
规则由模型配置决定。VQVAE 与 SAR 可共用相同协议的统计文件。

若要复现**确实使用旧的归一化角度训练的权重**，训练和采样配置必须保持一致，
可显式指定历史协议：

```yaml
normalization:
  rotation_normalization: minmax
```

正常训练无需填写此项，默认 `auto` 即上述模型规则。旧权重不会因为改动
normalizer 自动变成原始弧度的 sin/cos 模型；新协议需要对应训练的权重。
此修改不改变原始 NPZ 几何，不执行 AABB/OBB 转换。

验证（Python 命令位于 shell 脚本中，通过 Slurm GPU 作业运行）：

```bash
sbatch preprocess_script/verify_normalizer.slurm
```
