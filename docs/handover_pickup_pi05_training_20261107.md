# 从 ULA 接近状态训练 pi0.5 抓取红块

## 当前约定

按用户最新要求，训练对象是**原生 pi0.5**，不是 WorldLA/WAM；从本地 `external/pi05/pi05_base/params` 初始化，采用 LoRA 微调，总目标为 **100000 个优化步**。先让已启动的 10000 步实验完成，再恢复其权重与优化器，继续 90000 步；不是重新开始，也不是在 WorldLA 的 100000 步上继续编号。
只更新原生 pi0.5 的 LoRA 参数，不包含阶段路由器或消息适配器。原模型及其他任务代码不覆盖。
先前 WorldLA 两步诊断已结束，其输出仅为诊断文件，不作为本次初始化或结果。

## 50 条短轨迹

任务是 `handover_block` 的随机化场景，但仅学习左手抓起这一段，不包含双手交接和最终放置。

每轮先执行现有 RGB-D 识别和传统接近。从机械臂实际到达的接近状态开始记录，由传统几何控制执行张爪、向前伸入、闭爪、抬升 10 cm、保持。抓取目标由同一视觉接近位姿推算，不读取实时红块真值作规划。
仿真红块位姿只进入独立核验：红块整体离开初始支撑高度至少 3 cm，持续满足几何持握判据，末态仍被保持抬起。失败轨迹保留诊断记录，但不作为成功示范训练。

采集 seed 从 10000 开始，123 次尝试获得 50 条核验通过的示范。其余尝试分别为 56 次定位无有效输出、12 次场景初始化失败、2 次传统接近不到位、3 次抓起核验未通过。这是筛选成功示范的采集统计，**不是方法成功率评测**。

共 3688 个观察—动作样本。50 条成功轨迹末态最低面上升范围约 9.83–10.04 cm。
按固定随机种子分为 45 条训练、5 条验证，整条轨迹只属于一个集合，不将相邻帧随机拆分。

## 记录和训练输入

每个样本保存动作区间执行前的三路 RGB-D、机器人命令关节状态，以及区间末端的关节目标。不能把动作执行后的图片与同一动作配对。
每 15 个物理步采样，另外记录实际物理步边界。LeRobot 中沿用原数据的名义 15 fps，真实物理时间以原始 HDF5 步边界及仿真 250 Hz 为准。

本地 RoboTwin 的 `joint_action.vector` 读取的是关节驱动目标与夹爪控制值，并非实际关节反馈。为与现有部署接口保持一致，模型输入沿用该语义；实际机械臂关节反馈另外存入审核字段，不混为同一个输入。
模型输入只有三路 RGB、命令关节状态和语言；深度、相机标定、物体真值核验和额外反馈只保留作审核，不输入 pi0.5。

语言固定为：

> Pick up the red block with the left arm and hold it above the table.

## 原生 pi0.5 微调配置

- 初始化：本地 `pi05_base`，不加载 WorldLA 的动作权重。
- 模型：原生 pi0.5，PaliGemma 与动作专家均配置 LoRA；仅 LoRA 参数可训练，共 49,987,584 个。
- 数据：45 条训练 / 5 条验证；归一化仅用训练轨迹计算。
- 动作：沿用原生 Aloha 变换，机械臂关节使用相对当前命令状态的动作块增量，夹爪保持绝对控制值。
- 预算：总计 100000 个优化步，分为 0–10000 和 10000–100000 两段；全局 batch size 8，使用 GPU 0、1、2、3 共 4 张卡做 FSDP。
- 学习率：前一段峰值 1e-4，前 200 步预热，10000 步内余弦下降到 1e-5。续训保留原衰减时间范围，10000 步之后保持 1e-5，不重新预热或把学习率抬回峰值。
- 每 1000 步完整验证并保存原生检查点；检查点按实际完成的优化步编号。
- 未开启外部实验平台上传；所有新数据、日志、缓存、模型写在 ULA 下。

100000 步和 50 条数据是用户指定的本轮实验预算，不代表已经证明数据足够或不会过拟合。更长训练可能过拟合，验证损失和最终未见场景抓起率需要分别检查。

## 文件与启动

采集结果：`/data1/xuhuixin/ula/results/handover_pickup_collect50_20261107_v1/report.json`

数据集：`/data1/xuhuixin/ula/datasets/handover_pickup50_20261107_v1/dataset.json`

首段四卡训练目录：`/data1/xuhuixin/ula/checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1`

续训到总计 100000 步的目录：`/data1/xuhuixin/ula/checkpoints/pi05_handover_pickup50_100000_4gpu_20261107_v1`

最初的两卡训练在约 300 步时按用户要求停止，尚未到第一次 1000 步检查点保存。其日志与中断记录保留在原目录；四卡实验从相同 `pi05_base` 重新开始，总预算仍为 10000 步，不将未保存的两卡步数算进新模型进度。

```bash
cd /data1/xuhuixin/ula
GPU_LIST=0,1,2,3 bash scripts/train_pi05_handover_pickup.sh \
  --dataset /data1/xuhuixin/ula/datasets/handover_pickup50_20261107_v1 \
  --output /data1/xuhuixin/ula/checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1 \
  --steps 10000 --batch-size 8 --fsdp-devices 4 \
  > /data1/xuhuixin/ula/results/pi05_handover_pickup50_10000_4gpu_20261107_v1.log 2>&1
```

非空目录默认拒绝覆盖；新实验更换目录。中断后同一配置可加 `--resume`，从已经完整保存的检查点继续到总计 10000 步。
`protocol.json` 记录实际完成步数、模型结构开关、训练验证 seed、最新完整检查点和源文件哈希；`train_metrics.jsonl` 与 `validation_metrics.jsonl` 分别记录训练及验证损失。

```bash
tail -f /data1/xuhuixin/ula/results/pi05_handover_pickup50_10000_4gpu_20261107_v1.log
```

## 已安排的 100000 步续训

续训入口独立于正在运行的首段源码，不修改、停止或提前截断首段训练。
`scripts/continue_pi05_pickup_training.py` 等待首段状态为 `finished`、10000 步参数/优化器/归一化资产均保存完整、源进程退出后，再启动续训。
10000 步检查点通过硬链接复制到新的续训目录；原目录不参加后续滚动清理，因此仍保留可用于比较的 10000 步模型。

恢复的是完整训练状态，包含模型参数、优化器动量和优化步计数。训练样本打乱迭代器会重新初始化，因此不宣称与完全不中断训练逐位一致。
使用独立的 `scripts/train_pi05_handover_pickup_extended.py`，允许扩大总预算，但检查数据、batch size、学习率峰值及原衰减时间范围一致。

本轮已启动以下依赖任务（不要重复启动）：

```bash
cd /data1/xuhuixin/ula
external/pi05/.venv/bin/python scripts/continue_pi05_pickup_training.py \
  --source-run /data1/xuhuixin/ula/checkpoints/pi05_handover_pickup50_10000_4gpu_20261107_v1 \
  --output /data1/xuhuixin/ula/checkpoints/pi05_handover_pickup50_100000_4gpu_20261107_v1 \
  --expected-source-step 10000 --target-steps 100000 --gpu-list 0,1,2,3
```

续训任务状态位于新目录 `continuation.json`。等待期间为 `waiting_for_source_completion`，启动恢复后为 `resuming`，达到总计 100000 步后为 `finished`；源训练异常退出时停止并记录错误，不静默从头训练。

```bash
tail -f /data1/xuhuixin/ula/results/pi05_handover_pickup50_100000_4gpu_20261107_v1.log
```

恢复流程已用独立诊断检查点验证：从第 2 步恢复权重与优化器，实际训练到第 3 步，并再次验证和保存。此诊断不是抓取成功率评测。

后续测试需要新的 seed（例如从 20000 开始），使用相同的传统接近与语言指令，不用“空手抬高 8 cm”提前截断。抓起评分仍独立进行，不能用真值指挥动作。
