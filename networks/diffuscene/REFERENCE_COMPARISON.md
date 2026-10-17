检查日期：2026-09-29。

本次比较我们实际完成的 `livingroom_uncond_100k` 运行与
`/home/cscgbnu/zhx/public_code/DiffuScene`（Git `d78a289`，工作区干净）的
`config/uncond/diffusion_livingrooms_instancond_lat32_v.yaml`。
参考仓库还提供 eps、bedroom、文本条件等配置，以下结论不把这些不同任务混为一谈。

结论：我们保留了原实现的扩散数学和 U-Net 主干，但改变了输入编码、训练数据、
数据增强、训练时长/学习率、形状特征分支和 IoU 正则，因此属于 layout-only 变体。
上一轮“新旧脚本对齐”是指我们仓库内部两条路径，不能据此推导与外部原实现等价。

本次进行了源码/AST 比较、参考数据装饰器的确定性小样本复现和全部训练数据统计，
没有对两个方案进行对照重训。机器可读结果：
[reference_comparison_audit.json](runs/livingroom_uncond_100k/reference_comparison_audit.json)。
以下“影响大/中/小”是依据机制和已有消融结果的判断，不是我们数据上的实测提升比例。

| 项目 | 参考实现 livingroom-v | 我们实际运行 | 影响判断 |
| --- | --- | --- | --- |
| 扩散与 U-Net | DDPM 1000 步，linear beta=0.0001→0.02，v prediction，dim=512，4 级 [1,1,1,1] | 相同 | 主体模型没有被替换 |
| 条件 | 无房间/文本条件；128 维可学习槽位 embedding | 相同 | 无条件任务能力没有因去掉 ResNet 而损失 |
| 类别数和编码 | 24 家具类+END=25；删除 START；one-hot 转 -1/+1 | 31 家具类+START+END=33；0/1 | 中到大：改变类别去噪尺度和 END 语义，不能混用原解码 |
| 输入维度/分支 | bbox8+class25+shape32=65 | bbox8+class33=41，无 shape 分支 | 中到大，尤其 CAD 选择、对称性与语义一致性 |
| 物体排列 | 每次取样随机排列所有布局属性 | 固定原始 objList 顺序 | 中到大风险：模型可学习源数据槽位偏好，缺少排列增强 |
| 旋转增强 | 场景固定四方向旋转，位置和 yaw 同步变换 | 无增强；归一化布局预先缓存 | 中到大风险：方向覆盖和泛化不同 |
| 训练长度 | 150,000 个完整 epoch | 100,000 次 optimizer update | 大：不是同一训练预算 |
| LR 衰减 | 初始 2e-4，每 20,000 epoch 减半 | 初始 2e-4，每 10,000 update 减半 | 大：我们较早进入很小 LR |
| IoU loss | 开启，含 alpha_bar(t) 和 0.1 权重 | 关闭 | 中等，直接减少物体交叠方面的训练约束 |
| 数据分布 | 3D-FRONT 缓存；livingroom 预处理类别和过滤；3–21 物体 | Levels2021 重建 OBB；当前 train 2–27 物体，类别更广 | 大的可比性差异；不等于数据更多就一定更好或更坏 |
| split | train+val 训练、test 验证 | train 训练、val 验证、test 保留 | 改变训练样本与评估口径；同名 loss 也不能直接比较 |
| padding | 21 槽，先转 cos/sin，再补 [0,0] 角度 | 27 槽，先补 yaw=0，再转 [1,0] | 小到中等，约半数槽为空，改变无效槽监督分布 |
| 尺寸 | 半边长 | 完整边长，配套拟合 stats | 单独倍数变化对 minmax 目标无本质影响；物理几何接口必须配套 |
| 精度 | FP32 | BF16 autocast，参数/Adam 仍 FP32 | 预计小于上述协议和目标变化；没有质量 A/B 证明等效 |
| 验证与恢复 | 验证噪声不固定；保存模型、Adam、epoch | 固定验证噪声并恢复 RNG；保存 step/样本位置/RNG/数据指纹 | 主要改善监控与可复现性 |

**源码相同部分的证据。** `diffusion_ddpm.py` 和 `loss.py` 分别与参考文件逐字节
相同。`denoise_net.py` 去掉 import 后 AST 相同，`get_loss()` AST 也相同。
参数维度变化会增删输入/输出分支，但没有改变主干数学。
Adam、weight_decay=0、batch=128、梯度裁剪10也一致。
我们的真实物体 yaw 直接计算 cos/sin，与参考实现相同；上一轮提到 minmax-yaw
指的是我们仓库历史 normalizer，不是外部 DiffuScene 的 cosin_angle 数据路径。

**学习率与预算。** 参考
[train_diffusion.py:221](/home/cscgbnu/zhx/public_code/DiffuScene/scripts/train_diffusion.py:221)
将 epoch 索引 `i` 传入 scheduler，内层遍历整个 loader；
`steps_per_epoch=500` 对应的循环已注释掉，不能拿这个配置数字计算实际更新次数。
参考首轮减半发生在 `20000 * ceil(N_ref/128)` 次更新后，总预算是
`150000 * ceil(N_ref/128)` 次更新。参考缓存数据不在本地，不能确认其实际 N_ref。

我们的 train=5,642，所以每 epoch=45 update，100k update 约 2,222 epoch。
第 50,001 次更新 LR=6.25e-6，第 100,000 次更新 LR=3.90625e-7，是初始值的 1/512。
仅用相同的 5,642 场景换算参考循环，会得到首减半90万次更新、总675万次更新；
这只是统一数据规模下的单位比较，不是作者实际实验的训练量。

当前 [loss_summary.json](runs/livingroom_uncond_100k/loss_summary.json) 显示
best val 在56k为0.02194077，100k为0.02243964；40–50k与90–100k窗口中训练loss
还在降、验证loss基本持平。因此当前证据不支持简单追加同一衰减方案就能明显改善生成。
这也不能单独证明过早衰减是唯一原因；缺少数据增强、目标差异和过拟合都可能参与。

**类别、空槽与损失。** 参考
[Diffusion 数据装饰器:906](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/threed_front_dataset.py:906)
明确执行 `*2-1` 并删除 START。我们
[LayoutDataset:66](/home/cscgbnu/zhx/code/ARScene/networks/diffuscene/layout_data.py:66) 保留0/1和START。
在相同类别维数下，两个不同类别的码字平方距离从8变为2；固定高斯噪声下，类别间隔
变小。它是不同的学习问题，但不是不能学习的非法编码，不能仅据此断言生成必然崩溃。

我们的损失是 `alpha_bar[t] * (MSE_bbox + MSE_class)`；参考启用 shape 和 IoU 时是
`alpha_bar[t] * (MSE_bbox + MSE_class + MSE_shape) + L_IoU`。
因此 loss 数字更低也不代表我们的生成质量高于参考方案。
参考 END 的符号过滤及 IoU 中 `end_score<=0` 的有效物体判断都以 -1/+1 为前提。
我们当前采样使用 END 类 argmax，与0/1目标配套；直接接回原解码/IoU则不配套。

训练数据共74,122个物体，27槽中的 **51.34%** 是空槽。
空槽角度 `[1,0]` 不会引起其他字段平移，但它与参考的 `[0,0]` 是实际目标差异。
保留 START 多出一个从不为正的类别，增加无效输出的可能性；现有1000次采样未报告
START错误，不能把它描述成已观察到的主要失败原因。

**增强与分布。** 参考
[Permutation:576](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/threed_front_dataset.py:576)
及 [factory:1006](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/threed_front_dataset.py:1006)
启用随机排列、四方向旋转。我们每个场景的归一化物体行只编码一次且不排列。
当前第一槽有1,956/5,642场景是 multi_seat_sofa，整体最常见类别则是 dining_chair；
这说明源数据槽位携带明显的排序信息，不应把固定顺序等同于无序集合增强。

参考 livingroom 预处理限制3–21物体，并采用24类词表。我们有52个训练场景超过21物体、
10个仅2物体，还包含参考livingroom词表之外的 nightstand 903个、kids_bed 43个、
single_bed 71个、double_bed 9个、dressing_table 1个。
这些可能是混合房间的有效物体，不自动构成错误；它们表明数据分布和任务确实不同。
参考过滤见
[datasets/__init__.py:112](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/__init__.py:112)。

**尺寸与 IoU。** 对相同盒子，若尺寸和边界统计同时乘2，minmax结果不变，
所以 full-size/half-size 本身不是训练退化的充分原因。参考实现用 `center±size`
计算IoU，当前完整边长应使用 `center±size/2`；重新启用还需匹配 END 编码、
统计量和旋转盒子的几何定义。不能只把 `loss_iou` 开关改成 true。

**论文能提供的影响量级。** 作者的 bedroom 消融 Table 3 给出：

| 方案 | FID（低更好） | 对称物体对数 Sym |
| --- | --- | --- |
| 完整方案 | 17.21 | 0.72 |
| 去掉 IoU loss | 17.93 | 0.64 |
| 去掉形状特征 | 18.40 | 0.50 |

即该实验中去掉IoU的FID约恶化4.2%，去掉shape约恶化6.9%，shape消融的对称对数
约减少30.6%。这些只说明组件有可观贡献，**不能当作我们的 livingroom/Levels2021
数据上的预期百分比，更不能将两个效应相加**。来源：
[DiffuScene 论文 §4.2 / Table 3](https://arxiv.org/html/2303.14207v2#S4.SS2)。

**参考仓库自身的形状特征错位。** 当前 livingroom-v 配置的 encoding_type 是
`cached_diffusion_cosin_angle_wocm`，没有 `objfeats` / `lat32` 子串。
factory 只将类别、位置、尺寸、角度放入 permute_keys，缓存加载器却仍加载
objfeats_32，模型配置也要求32维特征。确定性复现：布局顺序变成 `[2,0,1]` 时，
特征顺序仍是 `[0,1,2]`。它会破坏同一物体几何与shape特征的对应关系。
证据：[factory:1037](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/threed_front_dataset.py:1037)、
[cached loader:369](/home/cscgbnu/zhx/public_code/DiffuScene/scene_synthesis/datasets/threed_front.py:369)。
我们 objfeat_dim=0 不受影响；如果以本地参考代码跑对照或恢复形状特征，应先解决。
这不证明论文作者的实际训练有该问题，论文实验配置可能与当前提交不同。

建议的验证顺序：在相同数据、验证集、生成评估和更新预算下，先分别对照学习率时间尺度、
旋转/排列增强、类别/空槽协议，再增加与当前几何匹配的IoU及真正对应的shape特征。
用独立生成样本衡量类别/物体数量分布、交叠、对称性、FID/KID及覆盖率；不以总去噪loss
单独判断生成质量。现有1000样本995个通过数值/正尺寸检查只证明格式有效，未测出无碰撞
或接近参考方案的视觉质量。
