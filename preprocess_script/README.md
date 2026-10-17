# Levels2021 上游预处理

入口是项目根目录的 `preprocess.py`，核心对象编码为 `utils.embed_obj_token()`，坐标转换共用 `preprocess_script/geometry.py`。输出保持原有 `room_data.npz`、`room_mask.png` 结构，可供原数据加载器和 RelTree 使用。

`obj_tokens` 保存 `[类别 one-hot, OBB 中心 xyz, 局部完整边长 xyz, yaw, 可选 latent]`。米制、右手系、y 向上、角度为弧度；ATISS 模式仅在类别段增加两个保留槽，尺寸仍是完整边长。房间水平中心移到原点，原始 `Levels2021` 文件不改写。

旧实现直接拼接资产参考点 `translate`、世界 AABB 尺寸和 yaw。新实现按 `T @ R @ S` 计算 OBB 中心，尺寸来自资产局部尺寸乘实例 scale 的绝对值。斜放 mesh 的 AABB 可以比旋转后局部盒子的 AABB 更小；只检查原始 AABB 是否被后者包含，不要求尺寸相等。

资产中心优先读取元数据中的 `center` / `loc`。当前元数据缺少这些字段时，只从训练集的轴对齐实例恢复资产局部中心，并保存 `asset_geometry.json`。这一步无梯度，但依赖训练数据标定。验证、测试实例不会更新共享资产中心。无法恢复中心、旋转带有倾斜、坐标非有限值或包含关系检查失败时，报告具体场景与原因，不猜测中心或静默删除物体。

当前卧室流水线保留旧数据的场景名单及 split，重新读取全部原始几何；采用 3 mm 的资产局部中心一致性容差（训练集审计最大偏差约 2.33 mm），以及 3 cm 的世界 AABB 包含检查容差。房间高度使用显式条件 3 m，不从家具高度反推。未传 `--split-root` 时使用固定 seed 的房屋级 70/10/20 划分；传入时保留已有划分。

```bash
mkdir -p networks/reltree/slurm
sbatch preprocess_script/preprocess_levels.slurm \
  --raw-root datasets/Levels2021 --filter_fn bedroom \
  --split-root datasets/processed_bedroom_wo_lat \
  --asset-center-tolerance .003 \
  --output datasets/processed_bedroom_wo_lat_v2
```

`--output` 指定完整目录；兼容原来的 `--out_dir` 自动追加房间类型和 `_wo_lat` 的行为。输出必须为空，防止不同定义的数据混用。每个 NPZ 都包含 `geometry_schema`、类别顺序、米制房间边界和 room frame；数据集完成后写入 `manifest.json` 和 `READY`。`preparation_report.json` 记录成功、失败、未匹配的旧场景及校准来源。

随后将上述 NPZ 编码成 RelTree：

```bash
sbatch networks/reltree/prepare_reltree.slurm \
  --processed-root datasets/processed_bedroom_wo_lat_v2 \
  --output datasets/processed_reltree_bedroom_v2
```

此阶段直接读取修正后的 `obj_tokens`，不再绕过上游重读原始家具几何。资产 ID 只保存在上游诊断字段中，不进入 RelTree token/query。旧数据和旧训练 checkpoint 保留；新数据使用独立目录和内容指纹，新训练不会混用旧 checkpoint。

2026-09-25 实测：Slurm 作业 16214 从 Levels2021 重建了全部 19,317 个已有卧室候选，接纳 train 12,999 / val 1,896 / test 3,724 个场景。旧错误检查拒绝的 30 个资产 8988 场景全部恢复。剩余 698 个失败场景中，697 个缺少可验证的资产局部中心，1 个实例缩放退化；均保留原因，未猜测缺失坐标。资产中心来自训练集恢复的 97,465 个物体实例和逐实例轴对齐恢复的 191 个实例，中心一致性冲突为 0。56 项 GPU 作业回归测试（16213）通过。
