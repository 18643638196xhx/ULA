# CAD 表面距离修复与大垃圾桶任务复测

日期：2026-11-11。

## 本轮只改什么

把 CAD 的“最近采样点距离”改为“最近三角面距离”。不修改相机视角、检测/分割候选、FoundationPose 配准、模型尺度、多实例分配，以及任何接受门槛；不执行机器人任务动作。

原先每个 CAD 随机采样 12000 点。大物体的表面采样较稀，点云到最近采样点的距离包含离散化误差，不能直接当作点云到模型表面的距离。

现在缓存每个静态 CAD 的三角网格及 CPU BVH 空间索引。每个候选位姿只把当前测量点逆变换到 CAD 坐标系，查询最近三角面上的最近点，使用无符号欧氏距离。BVH 搜索使用归一化坐标，找到三角面后回到原始双精度 CAD 坐标计算最近投影及距离；不使用最近采样点近似。无须物体封闭或直立，点在实体内部也不会被当作表面距离为零。查询分批执行，不创建全部测量点与全部三角面的笛卡尔积。完全退化的网格拒绝，有效网格中的零面积三角面不参加距离计算。缺少网格查询依赖时拒绝，不静默退回采样点近似。

静态 CAD 仍由调用方明确提供；本模块不读取物体位姿、身份像素、接触信息或任务名称。

## 保持不变的检查

- 表面距离中位数不超过 8 mm。
- 至少 65% 的测量点距 CAD 表面不超过 15 mm。
- 随后的渲染深度、可见覆盖、遮挡、自由空间冲突及多候选歧义检查全部保留。
- 表面距离通过不代表 CAD 整体核验通过，更不代表机器人任务成功。

输出增加 `distance_method=exact_unsigned_point_to_triangle_surface`、三角面数量、退化面数量和 `sampled_model_points=false`，便于确认真正调用了新实现。

## 代码

- [精确表面距离](/data1/xuhuixin/ula/adapters/unified_vision/cad_geometry.py)
- [FoundationPose 核验接入与模型缓存](/data1/xuhuixin/ula/adapters/unified_vision/cad_backend.py)
- [新增测试](/data1/xuhuixin/ula/tests/test_unified_vision_surface_distance.py)

## 验证

初版 CPU R-tree 查询的专项测试 62 项通过、全量测试 655 项通过，但真实冒烟中出现慢查询超时，因此未用于正式复测。慢版日志保留，不把超时当作定位失败率。最终版改用缓存 CPU BVH，并增加大坐标偏置精度测试；最终专项 **63 项通过、全量 656 项通过**。测试覆盖大三角形内部点、大小模型相同毫米误差、不同网格细分密度、刚体变换、边缘/顶点距离、实体内部点、原接受门槛、原始网格被调用者修改、随机状态不改变、非法/退化输入，以及和独立暴力三角面查询的一致性。

最终真实链路冒烟不再发生超时。两个初始样本的联合定位仍未通过，不能把“精确表面距离已接入”解释为两个任务已解决。正式复测保持所有配置冻结，进一步统计大桶输出、错选和联合定位的情况。

- [最终全量测试日志](/data1/xuhuixin/ula/results/unified_vision_exact_surface_tests_20261111_v2_bvh.log)
- [最终两个任务各一场真实链路冒烟报告](/data1/xuhuixin/ula/results/unified_vision_exact_surface_bins_smoke_20261111_v2_bvh/report.json)
- [保留的慢版冒烟报告（含超时，不作为定位率）](/data1/xuhuixin/ula/results/unified_vision_exact_surface_bins_smoke_20261111_v1/report.json)

## 正式复测设置

`dump_bin_bigbin` 与 `put_bottles_dustbin` 各 50 个有效 randomized 初始场景，均从 seed 0 开始；环境初始化拒绝另计并继续后续 seed，与上一轮一致。只使用初始头部 RGB-D，提供本轮目标的具体 CAD。预测保存锁定后才读取独立标签评分，评分不回传算法。

联合定位仍要求本场所有目标定位通过；同时单列大垃圾桶及其他目标的定位结果，避免把“大桶有改善”误报成整场通过。

上轮基线目录保留：

[前三项修复后的 2500 场结果](/data1/xuhuixin/ula/results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3/report.json)

本轮新目录：

[精确表面距离：两任务各 50 场](/data1/xuhuixin/ula/results/unified_vision_exact_surface_bins_randomized_50per_task_20261111_v1/report.json)

这些是已知 CAD 条件下的初始视觉定位实验，不是完整机器人任务成功率。相机视野严重截断的问题本轮没有一并修改，便于独立判断距离计算修复的效果。
