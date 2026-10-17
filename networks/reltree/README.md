# RelTree：功能关系构树、免训练 tokenizer 与 next-scale 生成模型

构树和相对 tokenizer 核心只依赖 NumPy，不导入 PyTorch、资产文件或项目的 `utils.py`。生成模型、数据打包和训练入口依赖 PyTorch，覆盖 Layout 阶段；不包含 cuboid 提取。

新增 `room_root_ar_v3`：**SOS 作为主 Transformer 序列中的 token，与边界条件使用同一个 self-attention，先预测 Room 的绝对长宽高，再生成 SPLIT/OBJECT**。边界是独立 condition；无条件生成用 null embedding，不需要外部 room frame，不输出边界多边形。后续沿用 query 与 shifted 节点的特征拼接、“平台＋阶梯”mask 和两层属性自回归。详见 [ROOT_AR.md](ROOT_AR.md)。原有已训练 `node_attribute_ar_v2` 的实现和权重仍可加载，历史说明见 [MODEL.md](MODEL.md)。

## 两个独立部分

1. `FunctionalTreeBuilder`：从类别、几何规则推断功能关系，形成分组，再构造二叉树。
2. `RelTreeTokenizer`：父子相对几何 → 固定标量 bin → 显式整树解码，无神经编码器、VAE 或学习式码本。

关系是启发式推断结果，不是真实功能标注。生成器的 embedding、Transformer 和输出 head 仍需训练。

## 快速使用

从项目根目录运行示例与测试：

```bash
python -m networks.reltree networks/reltree/example_scene.json --output /tmp/reltree-example.json
python -m unittest discover -s networks/reltree/tests -p test_reltree.py -v
```

Python API：

```python
from networks.reltree import Box3D, SceneObject, FunctionalTreeBuilder, RelTreeTokenizer

room = Box3D(position=(0, 1.5, 0), size=(8, 3, 6))
objects = [
    SceneObject("bed", Box3D((-2, .5, 0), (1.8, 1., 2.2))),
    SceneObject("nightstand", Box3D((-3.2, .3, .6), (.5, .6, .5))),
]
tree = FunctionalTreeBuilder().build(objects, room)
tokenizer = RelTreeTokenizer(categories=["bed", "nightstand"])
tokens = tokenizer.encode(tree)
reconstructed = tokenizer.decode(tokens)
report = tokenizer.reconstruction_report(tree, tokens)
batch = tokenizer.pad_batch([tokens])
tokenizer.save("/tmp/reltree-tokenizer.json")
```

默认类别顺序与项目 `THREED_FRONT_CATEGORY` 的 31 类一致；新类别必须在训练前显式加入共享 vocabulary。不要为每个场景重新建立类别编号。`object_id` 只用于诊断，编码、排序与 query 不使用资产身份。

## 几何输入协议

- 世界坐标以米计；右手系，y 向上，yaw 为弧度。
- `R_y(theta) = [[cos,0,sin],[0,1,0],[-sin,0,cos]]`。
- `position` 必须是 OBB 中心；`size` 是物体自身坐标轴上的完整边长，严格为正。
- 数据编码时 `room` 是场景的规范坐标框，不从 GT 家具反推。v3 生成时它由 SOS 预测；v2 生成时是外部给定条件。
- 默认 `min_size=1e-4 m`；过小输入和解码结果显式报错，不静默裁剪。
- 不规则房间的多边形可以由下游条件编码器使用；这里的 `room` 只是规范坐标框，不代表多边形占用。

旧 `utils.embed_obj_token` 拼接 `obj.translate` 和 `bbox.max-bbox.min`，无法保证符合上述 OBB 协议。现在由根目录 `preprocess.py` 修正上游表示，并为 NPZ 保存 `asset_local_obb_trs_v2` schema；同时修正了电视柜与多人沙发的类别映射。RelTree 直接读取这些修正后的 NPZ，拒绝未标记的旧格式。`objects_from_layout_tokens(..., geometry_convention="obb_center_full_size")` 适配未归一化的 `[one-hot, p(3), s(3), yaw, ...]` 数组，忽略尾部 latent；不从世界 AABB 反求 OBB。完整流程见 [上游预处理说明](../../preprocess_script/README.md)。

Livingroom 可直接复用修正后的 `datasets/atiss_livingroom_obb_v2_wo_lat`，
适配器根据 `class_token_count` 读取几何，去掉 ATISS 的两个预留类别槽。
构树及固定量化范围与卧室一致，不拟合验证或测试数据：

```bash
sbatch networks/reltree/prepare_livingroom.slurm
```

输出为 `datasets/processed_reltree_livingroom_v2`，包括 train/val/test 三个划分；
通用转换入口新增 `--include-test`，默认仍只转换 train/val。
`READY` 表示转换完成，`VALIDATED` 表示后续核验完成。
`preparation_report.json` 记录转换失败及裁剪，`validation_report.json` 记录
所有场景的解码检查、默认 SOS→Room 尺寸范围适配情况，以及每个划分
均匀抽取最多 64 个场景的完整父链重建误差（分别报告真实根框和量化根框）。
沿用上游 split；同一房屋跨 split 的情况会记录，不自动重新划分。

2026-09-29：作业 16417 完成 livingroom 转换与校验，train 5,638 / val 810 /
test 1,633，共 8,081 个场景，整树解码失败为 0，最大 27 个物体、树深度 12。
4 个训练场景因房间宽度只有约 0.00005 m，低于最小尺寸 0.0001 m 而被拒绝；
不是删除其中的家具。每个划分抽取 64 个场景，量化后对象中心平均误差分别为
3.65 / 3.91 / 3.53 cm（真实 Room 根框）；全部房间均在现有 root v3 的
16×8×16 m 尺寸范围内。作业 16418 的 `outlier_report.json` 记录了异常：
验证集一个 ceiling_lamp 中心高度为 5.319 m，超出固定 3 m 房高，相对位置
y 被裁剪，重建位置误差约 2.325 m。保留该样本和裁剪标记，不据此调整量化范围。
原有划分有 4 个 train/val 共享房屋和 1 个 train/test 共享房屋，没有重复场景 ID。

Livingroom 的 root v3 训练配置是
`training_configs/reltree_livingroom_root_ar.yaml`：主干 12×768、FFN ×4，
属性分支 2×256、FFN ×4，batch 128、BF16、cond_drop=0.1、100,000 steps。
5,638 个训练场景对应每 epoch 45 steps，epoch 上限设为 3,000，避免到 45k step
提前结束。输出目录为 `runs/livingroom_root_ar_12x768_attr256_cd10`，独立初始化，
不加载卧室权重；再次启动同一目录会恢复 `last.pt`。

```bash
# 单次分配一张 GPU，最长 48 小时。
sbatch networks/reltree/train_livingroom_root.slurm
# 或在 tmux 中使用监控脚本，超时/抢占后自动重新提交并续训。
tmux new-session -d -s reltree-living-12x768 \
  -c /home/cscgbnu/zhx/code/ARScene \
  'bash networks/reltree/run_livingroom_root_pipeline.sh; exec bash'
```

2026-09-29 已通过 tmux `reltree-living-12x768` 提交 Slurm **16419**，
使用一张 A800 80GB；参数量 91,899,937。每 100 step 及 epoch 末保存检查点，
每 1,000 step 验证。两种启动方式选择一种，避免重复提交。

## 功能聚类和构树

默认规则包含床—床头柜、床—衣柜、餐桌—椅子、书桌—椅子/书架、沙发—茶几/边桌/扶手椅/电视柜、梳妆台—椅子。类别先映射为规则 family，输出类别保持原标签。

语义关系要求水平 OBB 间距和垂直中心差处于阈值内；分数结合语义强度、间距衰减，以及可选的局部侧向/前后方向偏好。距离同时受米制阈值和房间水平对角线比例限制。方向规则不假设资产“正面”的正负号。支撑使用旋转矩形的实际交叠面积与顶部/底部高度差，不能据此宣称物理稳定性。

关系边按分数确定性排序并合并分组。同一组最多一个主要锚点（床、餐桌、书桌、沙发或梳妆台），避免不同活动区被共享邻近物体串联。这也意味着两张沙发默认不会成为同一个功能组；无法表示一件家具同时属于多个组。无关系物体保留为单独分组。

每组内部先根据归一化中心距离与包围体积增长合并成二叉树，再在场景层合并各组。所有 SPLIT 框均用子框完整角点在 room 坐标中求包围范围。左右顺序采用父局部中心字典序，平局使用由几何/类别确定的成员序；浮点主排序键保留 10 位小数，合并分数保留 12 位，仅用于平局处理。规范物体排序在近似键相同时继续比较精确几何，避免依赖输入顺序。完全相同的物体被视为几何上可交换的实例。

`GroupingConfig` 提供规则、family/anchor override、阈值和合并权重。`PairRule` 可新增关系。配置由 `to_dict()` 保存，也可通过 CLI 的 `--grouping-config` 加载。当前是小型室内场景的参考实现：穷举合并最坏约 O(N³)，不宣称大场景预处理加速；树可能不平衡。

`tree.groups` 与 `tree.relations` 的索引均指向 `tree.objects` 的规范顺序。`tree.diagnostics()` 记录深度、宽度、兄弟间距和越出房间框的物体数。它们是离线分析信息，不得作为早期生成 query 的输入。

## 编码与结构协议

节点按 BFS 打包，Room 位于 0，编码包含：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `types` | `[L]` | SPLIT=0, OBJECT=1, EMPTY=2, ROOM=3；批次 PAD=4 |
| `geometry` | `[L,7]` | 归一化位置 xyz、相对 yaw、逐轴 log 尺寸比的 bin |
| `classes` | `[L]` | 仅 OBJECT 有类别，其余 -1 |
| `parents/depths/slots` | `[L]` | 拓扑；Room 的 parent/slot 为 -1 |
| `clipped` | `[L,7]` | 超出非周期量化范围的标志，供离线统计 |

普通 N≥2 场景含 Room 共 `2N-1` 个节点；只有一个物体时使用 `(OBJECT, EMPTY)` 根协议；N=0 拒绝。SPLIT 恰有两个非空子节点。解码验证类型组合、父索引、槽位、类别和 bin，不接受非法结构。编码不含 asset ID、功能组标签或未来子树大小。

相对位置为 `R_y(-yaw_parent) @ (p_child-p_parent) / size_parent`，相对 yaw 环形 wrap，尺寸为 `log(size_child)-log(size_parent)`。解码逐节点使用**已解码的父框**，不用 GT 父框。无量化接口为 `relative_geometry` / `decode_relative`。

本版本明确采用一个约束：**SPLIT 相对 yaw 恒为精确 0**。序列中仍保留固定 yaw bin，但不预测或监督这个字段，解码强制为 0。OBJECT yaw 正常量化。这样内部框不会因为 bin 中点偏差逐层旋转。

## 量化、标定和重建

默认位置范围为每轴 `[-0.5,0.5]`，尺寸 log 比范围 `[-8,2]`，yaw `[-pi,pi)`；bin 数 `(256,256,256,255,256,256,256)`。默认 yaw 使用奇数个 bin，使零恰为一个中心。尺寸允许正 log 比。以上是原型默认值，不是已验证的最优配置。

非周期标量按均匀区间取整，解码到区间中点；yaw wrap 后使用环形 bin。越界值会裁剪并记录标志，`clipping_report` 输出比例。`reconstruction_report` 按完整父链解码，报告各深度的米制位置误差、环形角误差和尺寸相对误差。相对编码不自动保证预测的父子包含、贴地、无碰撞或正确支撑。

若希望标定范围，必须显式提供训练树：

```python
calibrated = tokenizer.calibrate(training_trees, split="train", quantiles=(0., 1.), margin=.01)
calibrated.save("/tmp/reltree-calibrated.json")
```

验证/测试阶段只加载该 manifest。函数拒绝 `split="val"/"test"`，但无法自动判断调用者传入数组的真实来源。标定仍无梯度训练，但依赖训练数据。manifest 保存类别、范围、bin、坐标协议、标定来源和样本数；编码附带 schema 指纹，防止错用配置。CLI 输出还保留分组参数、裁剪统计和解码失败，不丢弃失败样本。

## 接入后续生成器

`pad_batch` 返回 NumPy 目标数组、有效/padding mask，以及 type/layout/class loss mask。Room、padding 无监督；EMPTY 只有 type；类别只监督 OBJECT；固定 SPLIT yaw 无监督。调用方应按场景平均有效 query loss，再做 batch 平均，空类别集合取零。

当前或未来节点的完整目标不能用来初始化当前 query。同层前一个节点的 raw embedding 可以作为 shifted 输入与当前 query 在特征维拼接；每层第一个节点使用 SOS。`make_child_queries([(parent_index, known_type, known_path), ...])` 只根据已知历史为 Room/SPLIT 各产生两个槽位，OBJECT 不展开。位置特征使用 `path/depth/slot`，不使用依赖全部 GT 长度的打包下标。

生成模型位于 `model.py`，实际 teacher forcing 打包位于 `packing.py`；训练时应使用 `RelTreeCollator`，而非直接把上述目标数组喂给模型。`test_model.py` 和 `test_autoregression.py` 验证多层 mask 泄漏、padding 独立性、一次 forward 与逐节点/逐属性解码的一致性、动态推理和旧模型加载；`test_condition_dropout.py` 验证条件整体丢弃和验证时的条件保留；`test_resume.py` 还验证 batch size 变更后准确恢复未消费样本与优化器。75 项测试已在 GPU 通过：`sbatch networks/reltree/verify_resume.slurm`；真实数据短训、单场景串行复现及恢复验证：`sbatch networks/reltree/validate_ar.slurm`。

已有 `datasets/processed_reltree_bedroom_v2` 可以直接训练当前架构。全量训练入口为 `sbatch networks/reltree/train_full_reltree.slurm`，当前配置为 8 层 512 维、10% 场景级条件丢弃、单卡 batch size 128，累计训练 100,000 steps（包含断点中的已完成步数），输出 `runs/bedroom_ar_8x512_cd10`，自动从该目录的 `last.pt` 恢复。epoch 上限设为 1000，确保可以达到步数目标。旧并行模型的 `runs/bedroom_full_v2` 保留，不能直接作为新架构的训练断点。

更大模型配置为 `training_configs/reltree_12x768.yaml`：主干 12 层、768 维、12 heads、FFN ×4；属性分支 2 层、256 维、4 heads、FFN ×4；共 91,574,817 个参数。保持 batch 128、BF16、cond_drop 0.1、学习率 0.0003 和 100,000 steps，使用独立目录 `runs/bedroom_ar_12x768_attr256_cd10`。通过 `bash networks/reltree/run_large_pipeline.sh [前置作业ID]` 在 tmux 中运行；省略 ID 时读取 8 层运行目录中的最新作业 ID。流水线提交 `afterok` 依赖，成功完成前置训练后才分配 GPU；随后支持同目录 `last.pt` 自动恢复。2026-09-26 已在 tmux `reltree-ar-12x768-cd10` 中提交作业 **16288**，依赖 **16261**，无需重复提交。
