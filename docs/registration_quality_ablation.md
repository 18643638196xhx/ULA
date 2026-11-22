# 配准质量门槛放宽实验

## 目的与对照

检查当前被配准质量门槛拒绝的样本，放行后是否能完成真实搬运放置。
**配准通过率不是任务成功率**；放行后规划失败、执行失败或最终未满足任务标准都仍算失败。

直接对照是上一轮 `pointcloud + multiview + one grasp retry` 的完整 50 轮：
`results/pointcloud_mv_retry_50eps_seed0_gpu5_v1/report.json`，46/50（92%）。
不覆盖该基线；单独的两个失败 seed 试验不混入完整 50 轮分母。

## 唯一控制参数变化

| 配准门槛 | 默认/基线 | 本次显式设置 |
| --- | ---: | ---: |
| 裁剪后最大 RMSE | 8 mm | 12 mm |
| 最大估计倾斜 | 25° | 35° |

默认值不变，新增参数为：

```text
--registration-max-rmse 0.012
--registration-max-tilt 35
```

保持不变：ICP 优化过程、对应距离、体素大小、起点和迭代次数；双向重叠率门槛；
原点云处理；不启用机器人区域过滤；补观测、最多一次抓取恢复、VLA 总预算 500 步；
末端到位容差、规划碰撞配置、释放间隙与原任务成功判据。

原任务成功仅检查功能点 XY/Z 误差和夹爪打开，不检查精确直立角度。
因此新增 `evaluation_only.final_cup_tilt_from_initial_upright_deg`：
从仿真中的初始/最终旋转计算杯子原先竖直方向的最终倾斜。
该真值仅用于事后评测，不输入感知、配准、目标生成或放宽后的通过判断。
它不是新成功门槛，避免暗中改变与旧结果的比较口径。

程序报告 `pointcloud_registration_config` 和每轮实际使用的配置，
非默认门槛时不再将 `icp_quality_thresholds_unchanged` 标成 true。

## 判断“因放宽才获准”

报告每个 ICP 起点的几何误差和重叠度，可按原 8 mm / 25° 条件作事后对照。
需区分：

1. 本来就能通过原门槛的轮次。
2. 只因放宽才有候选通过的轮次。
3. 新门槛选择了原来会被拒绝的另一候选，即使原来已有别的可行配准。

放宽后参与排序的候选可能改变，所以不保证只是“多放行两个原失败样本”。
低残差也不证明物体位姿准确；对称、遮挡和机器人点混入仍可能产生误配准。

## 复现

先选择空闲 GPU，用新输出目录。当前参数只用于仿真对照，不等同于实机安全设置。

```bash
cd /data1/xuhuixin/ula
GPU=5 bash scripts/run_place_empty_cup.sh \
  --mode hybrid --approach-mode pointcloud \
  --pointcloud-multiview --grasp-retries 1 --grasp-stall-steps 120 \
  --registration-max-rmse 0.012 --registration-max-tilt 35 \
  --episodes 50 --seed 0 --max-vla-steps 500 \
  --output /data1/xuhuixin/ula/results/pointcloud_relaxed_50eps_repeat
```

72 项测试通过，覆盖默认保持 8 mm/25°、显式设置及无效参数拒绝、
改变门槛不改变相同点云的 ICP 优化指标，以及原有控制与拒绝执行行为。

## 完整实验结果

seeds 0–49 完整运行结束：48/50（96%），上一轮为 46/50（92%），无新增失败。
seed 33 因残差放宽而获准，seed 48 因倾斜放宽而获准，二者都完成原任务成功检查。
但 seed 48 最终仍倾斜约 5.56°，不能将任务成功等同于精确直立。
排除首轮，成功轮平均 5.895244 s，全部尝试平均 5.818729 s。
详见 [50 轮门槛对照报告](/data1/xuhuixin/ula/results/pointcloud_relaxed_12mm_35deg_50eps_seed0_gpu5_v1/SUMMARY.md)。
