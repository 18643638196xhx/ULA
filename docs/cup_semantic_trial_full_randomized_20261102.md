# 抓后识别修改后的随机化杯子完整任务评测

## 结论

`place_empty_cup`，`demo_randomized`，原始 seed 0–49。完整任务成功 **32/50（64%）**。其中 8 次场景初始化失败；仅以成功建立的 42 个场景计算，为 **32/42（76.2%）**。

平均成功完成时间为 **11.955 s**，排除整批第一轮 seed 0 后，对剩余 31 个成功轮次求平均。成功率仍使用全部 50 次，不因时间统计排除首轮而改变。时间包含环境建立、观察、推理、规划、执行及图片/深度记录，不是纯模型推理时间。

## 方法与范围

本轮沿用已知杯子模型方案：传统控制依靠视觉估计的已知 CAD 位姿完成接近，VLA 抓取，新局部语义分割和三段试抬确认抓取，原传统已知模型流程执行搬运、下降、松爪、撤离。不是无 CAD 的完整 pointcloud 放置方案。

新抓后识别不再硬筛蓝色像素，每段试抬后保存头部和腕部 RGB-D，并做同一相机前后配准。最终 3 cm、幅度比、方向与 ICP 门槛保持四样本复测设置。原接近、原放置和它们的视觉识别实现没有修改；最多一次抓取恢复，和原 50 次比较口径一致。模型权重、固定指令、固定左臂不变。这不是 RoboTwin 官方的随机 unseen 指令完整基准。

完整流程未接入诊断用的抓后仿真真值核验。抓取控制期间不读杯子真值或物理接触；仿真真值只在控制结束后用于原任务评分。检查确认没有生成中途 `evaluation_after_control.json` 文件。

## 和上轮比较

| 同为随机化、已知杯子模型 | 上轮 | 本轮 |
| --- | --- | --- |
| 全部 50 次完整任务成功率 | 33/50，66% | 32/50，64% |
| 成功建场后的成功率 | 33/42，78.6% | 32/42，76.2% |
| 平均成功时间，排除首轮 | 9.354 s | 11.955 s |

本轮没有新增完整成功的 seed，原先成功的 seed 3 本轮未成功。两批不是逐帧相同的抓取轨迹，不能仅凭一次 50 样本差异断言算法必然退化。新增分段双视角推理和记录会产生耗时；共享 GPU 状态、成功样本构成也不同，因此时间差不是严格的单因素测量。

四样本抓取诊断的 3 次通过不能转化成完整任务的 3 次成功。该诊断与本批的 VLA 执行也不完全相同。本批 seed 9 在抓后确认阶段失败，seed 32 虽通过抓取确认但最终放置评分未通过，seed 48 在放置前感知阶段失败。

## 18 次未成功的分布

| 阶段 | 次数 | seed |
| --- | --- | --- |
| 场景初始化失败 | 8 | 4、6、12、16、26、31、35、38 |
| 初始杯垫识别拒绝 | 1 | 19 |
| 试抬后未建立可信跟随 | 4 | 9、21、22、34 |
| 放置前杯子分割失败 | 1 | 48 |
| 放置动作返回成功，但最终任务评分失败 | 4 | 3、17、23、32 |

最后一组的最终杯底参考点与杯垫中心 XY 距离分别为 13.07、26.25、19.38、38.89 cm。它们证明最终没有放在评分要求的位置，但单凭终点距离不能确定是何时滑落、搬运偏差还是早先目标识别错误。seed 17 在此前离线核对中已确认将大盘子误认成杯垫；其余应结合本批过程观测进一步定位，不能统称为规划失败。

seed 48 的失败来自仍沿用的原放置感知分支；这说明只改善抓后确认，没有覆盖后续重新识别和放置本身的问题。本轮仅测量，不在运行中调参或修改放置实现。

## 记录与复现

- [完整报告](/data1/xuhuixin/ula/results/cup_known_semantic_trial_full_randomized_50eps_seed0_20261102_v1/report.json)
- [统计汇总](/data1/xuhuixin/ula/results/cup_known_semantic_trial_full_randomized_50eps_seed0_20261102_v1/evaluation_summary.json)
- [运行日志](/data1/xuhuixin/ula/results/cup_known_semantic_trial_full_randomized_50eps_seed0_20261102_v1.launcher.log)
- [独立完整评测入口](/data1/xuhuixin/ula/scripts/run_cup_trial_full_randomized.py)

运行期间保护代码与新分支代码哈希保持不变。视觉进程加载一次，批次结束后关闭。311 项回归测试通过。启动命令如下；再次运行必须更换输出目录，程序不会覆盖旧结果。

```sh
GPU=7 ULA_CUP_TARGET_FOCUS=1 ULA_CUP_VLA_FOCUS=0 \
ULA_CUP_RUNNER=/data1/xuhuixin/ula/scripts/run_cup_trial_full_randomized.py \
MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash /data1/xuhuixin/ula/scripts/run_place_empty_cup_randomized.sh \
  --mode hybrid --episodes 50 --seed 0 --approach-mode pointcloud \
  --grasp-handoff visual-trial-lift --contact-gripper-threshold 0.05 \
  --grasp-stable-steps 3 --registration-max-rmse 0.012 \
  --registration-max-tilt 35 --skip-trial-slip-check --grasp-retries 1 \
  --output /data1/xuhuixin/ula/results/cup_known_semantic_trial_full_randomized_50eps_seed0_20261102_v1
```
