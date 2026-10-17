本轮为从头训练的 livingroom 无条件 DiffuScene，配置：
`training_configs/diffuscene_livingroom_uncond_900k_aug.yaml`。

按本次指定的 optimizer update 预算运行900,000步，初始学习率2e-4，每完成120,000步
减半。第120,001步使用1e-4，第240,001步使用5e-5，最后阶段使用1.5625e-6。
该预算采用显式step数；当前5,642个场景、batch128每epoch有45步，900k步对应
当前数据的20,000个epoch，不会将150k再乘以当前loader长度。

每个训练场景、每个epoch均随机选择0°/90°/180°/270°之一（等概率），并随机排列
真实物体。旋转在归一化之前的物理坐标系中进行，中心位置和yaw同步变换，局部OBB
完整边长保持不变。类别、位置、尺寸、yaw作为同一行一起排列，最后添加END padding。
验证集保持原始方向与物体顺序。

训练集位置统计扩展为四个旋转方向的并集：x/z范围共同为`[-R,R]`，R只由训练物体
中心的x/z最大绝对值决定。它避免旋转后超出旧的非对称范围而被clamp。
y和尺寸统计仍由真实训练物体拟合。采样使用新checkpoint自带的normalizer。

增强的随机种子由训练seed、epoch、场景index确定，不消费模型扩散噪声的随机状态，
也不依赖DataLoader预取位置。增强协议记录在数据审计与checkpoint恢复协议中。
未设置augmentations的旧配置继续使用原有数据编码与统计，旧100k运行保留。

模型维度、0/1类别编码、33类（含两个保留槽）、IoU关闭、objfeat关闭等沿用当前
layout-only方案；本轮改动集中在预算、学习率、增强及其归一化/恢复兼容。
使用dim512、batch128、BF16，容量27，1000步v-prediction扩散，Adam、梯度裁剪10。

GPU验证：

```bash
sbatch --export=ALL,DIFFUSCENE_CONFIG=training_configs/diffuscene_livingroom_uncond_900k_aug.yaml \
  --job-name=diffuscene-verify900k networks/diffuscene/verify_livingroom.slurm
```

验证包括合成场景OBB角点刚体旋转、类别/几何对应、padding、增强可复现、全量训练
物体四方向归一化往返、验证集不增强、学习率边界、BF16连续/恢复训练精确一致，
以及实际宽度和batch的完整train/val训练与评估。

验证成功后启动持久监控：

```bash
tmux new-session -d -s diffuscene-living-900k-aug \
  -c /home/cscgbnu/zhx/code/ARScene \
  'bash networks/diffuscene/run_livingroom_900k_pipeline.sh'
```

监控脚本每次向Slurm申请一张GPU、48小时；抢占、超时或安全中断时自动提交下一次
分配并从last.pt恢复。正常错误、手动取消不会自动重启。运行到900k步写COMPLETE。
每20步记录指标、每500步保存last.pt、每1000步评估完整val并更新best.pt。

运行目录：`networks/diffuscene/runs/livingroom_uncond_900k_aug/`。

```bash
tail -f networks/diffuscene/runs/livingroom_uncond_900k_aug/metrics.jsonl
cat networks/diffuscene/runs/livingroom_uncond_900k_aug/train_job.txt
tmux attach -t diffuscene-living-900k-aug
```
