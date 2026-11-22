# 随机化杯子任务：取消额外试抬，50 个有效场景

## 结果

完整任务成功 **41/50（82%）**。从 seed 0 开始连续尝试到 seed 57，8 次初始化失败单独排除，补足 50 个成功初始化的场景。感知、抓取、规划、放置失败均保留在方法成功率分母中，没有只挑可操作或成功的场景。

平均成功完成时间为 **8.525 s**：排除第一个有效场景 seed 0 的加载/预热，对剩余 40 个成功轮次求平均。成功率使用全部 50 个有效场景，不排除首轮。

本轮使用已知杯子模型方案。取消的是额外验证用的 5 cm 试抬和视觉跟随门槛，不是取消任务正常需要的搬运、下降、松爪与撤离。

## 实际执行流程

传统视觉定位并接近 → VLA 操作 → 连续 3 个动作的闭爪命令低于 0.05 → 直接进入原传统搬运与放置 → 最终任务评分。

没有额外延迟 VLA 10/20 个动作，没有专门试抬，也没有在抓后读取物体位姿或物理接触真值来决定能否搬运。连续闭爪仅是交接许可，不是“已经夹稳”的证明。控制期间保留原有的最终评分屏蔽，仿真物体真值只在任务控制结束后用于官方评分。

接近与放置仍沿用原已知 CAD + FoundationPose 的视觉估计、接触/功能点标注和运动规划；其源文件未修改。原 VLA 权重、固定指令、左臂和最多一次抓取停滞恢复保持不变。没有将额外试抬换成另一种隐含测试。

49 次闭爪候选交接全部检查确认：试抬分段数为 0、视觉验证没有执行、没有试抬 RGB-D 文件。另一个有效场景 seed 19 在初始杯垫识别阶段失败，尚未进入 VLA。

## 与上一轮相同场景比较

| 共同的 42 个有效场景 | 有额外试抬 | 无额外试抬 |
| --- | --- | --- |
| 完整任务成功 | 32/42，76.2% | 36/42，85.7% |

新增成功 seed：3、21、22、23、34。新增失败 seed：13。共同场景净增加 4 次成功，约 9.5 个百分点。

另补测的 8 个有效场景为 seed 50–57，其中 5 次成功，3 次失败，合计得到本轮 41/50。82% 与上一轮 76.2% 的样本集合不同；判断变化应优先参考共同 42 场景的比较。

这是两批运行结果，不是逐动作完全相同的因果回放，不能据此声称所有失败都由试抬造成。VLA 实际轨迹仍可能有差异。本轮视觉与 VLA 分别在 GPU6 和 GPU7，使用 `platform` 按需显存分配；时间包含环境建立、推理、规划、执行和记录，不应当作严格硬件隔离的单因素速度对照。前 42 个有效场景排除首轮后的平均成功时间为 7.924 s；50 场景的 8.525 s 还包含补测批次的环境重建开销。

## 重点四个 seed

| seed | 本轮完整结果 |
| --- | --- |
| 9 | 失败，无法规划可行的搬运与下降组合，尚未执行放置运动 |
| 21 | 成功 |
| 32 | 放置动作执行结束，但最终位置不符合任务评分 |
| 48 | 放置前重新识别杯子时存在多个候选，拒绝选择 |

## 9 次有效任务失败

| 阶段 | 数量 | seed |
| --- | --- | --- |
| 搬运/下降组合规划失败 | 4 | 9、17、53、54 |
| 放置前杯子识别歧义 | 2 | 48、51 |
| 初始杯垫识别失败 | 1 | 19 |
| 动作执行完成但最终评分失败 | 2 | 13、32 |

最后两次的最终杯底参考点与杯垫中心 XY 距离分别约 20.96 cm、38.05 cm。这只能说明最后没有放对，不能仅凭终点断言是哪一刻滑落或何种估计出错。

8 个环境初始化失败 seed 为 4、6、12、16、26、31、35、38，与方法失败分开记录，不计入 50 个有效场景。

## 记录

- [完整报告](/data1/xuhuixin/ula/results/cup_no_trial_randomized_50valid_seed0_20261102_v1/report.json)
- [核验后的统计摘要](/data1/xuhuixin/ula/results/cup_no_trial_randomized_50valid_seed0_20261102_v1/evaluation_summary.json)
- [运行日志](/data1/xuhuixin/ula/results/cup_no_trial_randomized_50valid_seed0_20261102_v1.launcher.log)
- [独立无试抬入口](/data1/xuhuixin/ula/scripts/run_cup_no_trial_randomized.py)

本轮 319 项回归测试通过；运行期间原代码和新入口代码哈希不变。视觉进程整批只加载一次，99 次请求中 96 次正常返回、3 次感知拒绝；进程已关闭。感知拒绝不是模型进程崩溃，已包含在上述有效任务失败中。

再次运行时需更换输出目录。该入口复用原解析器的 `visual-trial-lift` 选项以保留闭爪触发及控制期间评分屏蔽，但已在独立入口中将试抬函数替换为零动作交接许可；实际没有执行试抬，报告已据实标注。

```sh
GPU=7 ULA_NO_TRIAL_VISION_GPU=6 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
ULA_CUP_TARGET_FOCUS=1 ULA_CUP_VLA_FOCUS=0 \
ULA_CUP_RUNNER=/data1/xuhuixin/ula/scripts/run_cup_no_trial_randomized.py \
MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash /data1/xuhuixin/ula/scripts/run_place_empty_cup_randomized.sh \
  --mode hybrid --episodes 50 --seed 0 --approach-mode pointcloud \
  --grasp-handoff visual-trial-lift --contact-gripper-threshold 0.05 \
  --grasp-stable-steps 3 --registration-max-rmse 0.012 \
  --registration-max-tilt 35 --skip-trial-slip-check --grasp-retries 1 \
  --output /data1/xuhuixin/ula/results/cup_no_trial_randomized_50valid_seed0_20261102_v1
```
