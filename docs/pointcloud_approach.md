# 无预备杯子模型：RGB-D 接近与 VLA 抓取验证

## 验证范围

新增 `--approach-mode pointcloud`。传统接近阶段不读取杯子 CAD/网格、
尺寸元数据、抓取接触点，也不加载或调用 FoundationPose。
它只需要 RGB-D、相机内外参、机器人运动学，以及通用检测/分割模型权重。
VLA 使用原有权重和输入，不重新训练。

最初验证“传统接近 → VLA 抓稳”，使用 `--mode grasp-only`，到抓稳就停止。
**该历史实验不是完整放置实验。** 后续已新增 `pointcloud + hybrid` 的无 CAD
视觉放置分支，见 [点云完整放置说明](/data1/xuhuixin/ula/docs/pointcloud_placement.md)。
`pointcloud + traditional/vla` 仍拒绝运行，避免改变纯传统/纯 VLA 模式的含义。
`--mode approach-only` 只验证接近到位，不加载 VLA。

RoboTwin 仍需要杯子资产来渲染和模拟物理；“不依赖模型”指控制器不读取
该资产作为物体几何先验，并非仿真场景不再需要物体。VLA 仍可能具有训练中
学到的杯子先验，本实验也不证明任意新物体泛化。

## 从图像到接近目标

1. GroundingDINO 用 `blue cup` / `round coaster` 提示检测，SAM2 分割像素区域。
2. 杯子掩码腐蚀一像素，减少轮廓处背景深度混入；有效点不足时只退回原掩码，
   不退回模型法。对有效深度回投，转换到世界坐标。
3. 在杯子周围图像区域、排除杯子和杯垫后，按世界 Z 高度众数估计水平桌面高度。
   去掉杯子点云中不高于桌面 3 mm 的点。
4. 各轴取 1% / 99% 分位数，得到**可见表面范围**。遮挡背面没有补全，
   这既不是完整物体包围盒，也不是物体 6D 位姿。
5. 左臂选可见范围的 -X 边、右臂选 +X 边，Y 取范围中心，
   TCP 高度取可见最高点 + 10 cm。使用原固定向下姿态和三个偏航候选。
6. 由机器人运动学将 TCP 目标换算为 EE 目标，CuRobo 先查询规划，
   首个可行候选只执行一次。实际 EE 到位误差需 ≤2 cm / 15°，才让 VLA 接管。
7. VLA 最后接近、闭爪、抬起；原仿真抓稳判定通过后立即中断动作块并结束本轮。

保留明确假设：水平桌面、世界 +Z 向上、已标定相机、当前双臂机器人和向下接近模板。
点数不足、桌面估计失败或规划失败会报告错误，不调用 FoundationPose 救场。
深度空洞、透明/反光杯子、严重遮挡和分割错误仍可能导致失败。
CuRobo 原有碰撞配置保持不变；本次没有新增基于观测点云的障碍地图，不能据此宣称
未知杂乱场景避障或实机安全。

抓稳检测仍使用仿真杯子位姿和接触信息：抬起 ≥3 cm、夹爪开度 <0.6、
有夹爪接触，连续 3 个动作相对位置变化 ≤1 cm。这些信号不输入 VLA 或接近目标生成，
但用于停止判定，所以还不是完全脱离仿真真值的系统。

## 运行与结果口径

```bash
cd /data1/xuhuixin/ula
GPU=5 bash scripts/run_place_empty_cup.sh \
  --mode grasp-only --approach-mode pointcloud \
  --episodes 10 --seed 0 --max-vla-steps 500 \
  --output /data1/xuhuixin/ula/results/pointcloud_grasp_repeat
```

GPU 编号只作示例，先检查空闲情况；使用新输出目录，不覆盖已有实验。
几何参数仍复用 `--geometry-top-clearance`、`--geometry-edge-fraction`、
`--geometry-yaw-offsets`，不使用标注模式的 `--pregrasp-distance`。

- `report.success_definition=stable_lifted_grasp_not_task_placement`：本次成功是抓稳，
  `task_success` 另存原 RoboTwin 放置检查结果，二者不能混同。
- `approach-only` 的 `success` 表示接近交接通过，同样不是放置成功。
- `handoff.estimated_cup_world_pose=null`：不捏造杯子姿态，另存观测范围。
- `handoff.geometry_approach.observation` 记录桌面高度、点数、裁剪比例、范围及依赖审计。
- `report.perception_worker.backend=pointcloud`，每批只加载一次 DINO/SAM2，不加载 FP。
- `bounds_target_generation` 是范围到目标的计时；`pointcloud_geometry` 是感知内部回投/过滤计时。
  原 `geometry` 的生成器会嵌套调用范围生成器，排他计时仍不重复相加。

耗时到抓稳即止，不能直接和此前完整放置 6.906 s / 6.978 s 做速度比较。
批次第 1 轮保留模型加载和预热开销，均值同时报告排除首轮的口径。

## 代码入口

- [rgbd_perception.py](/data1/xuhuixin/ula/adapters/rgbd_perception.py)：只加载通用检测、分割模型。
- [pointcloud_geometry.py](/data1/xuhuixin/ula/adapters/pointcloud_geometry.py)：由分割深度得到可见范围。
- [approach_geometry.py](/data1/xuhuixin/ula/adapters/approach_geometry.py)：范围到 TCP/EE 及规划执行。
- [run_place_empty_cup.py](/data1/xuhuixin/ula/scripts/run_place_empty_cup.py)：模式选择、抓稳终止和统计。

## 本次实际结果

GPU 5，seeds 0–9，接近到位 10/10，稳定抓取 10/10，无执行错误。
排除首轮后平均到抓稳耗时 **4.631805 s**，其中传统接近 0.484768 s，
VLA 抓取 3.410163 s，其余为建环境、截图和评测等。
全部在第一个接近候选成功，51 项测试通过。

这不是完整放置成功率：所有轮次在抓稳处停止，`task_success=false`。
详细设置、逐轮结果及分段时间见
[10 轮抓取报告](/data1/xuhuixin/ula/results/pointcloud_grasp_10eps_seed0_gpu5/SUMMARY.md)。
