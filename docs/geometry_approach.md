# 不依赖抓取接触点标注的几何接近模式

本实现只替换“传统接近目标如何生成”，VLA、夹爪准备开度 0.6、到位阈值、
抓稳交回和传统放置保持不变。它是适用于当前直立杯子、水平桌面先导实验的
几何规则，不是通用抓取检测器，也不是已验证的新物体泛化方案。

## 1. 两种接近模式

| 模式 | 目标来源 | 接触点依赖 |
| --- | --- | --- |
| `annotated`，保留为默认 | 视觉位姿 + 模型预置接触点 → grasp/pre-grasp | 使用 `contact_points_pose`，调用 `grasp_actor` |
| `geometry`，显式启用 | 视觉位姿 + 模型包围盒 → 上方偏侧 TCP 目标 → EE 目标 | 不使用抓取接触点，不调用 `grasp_actor` |

此前的 50/50 成功、6.906 s 报告对应 `annotated`，不能算作 `geometry` 的实验结果。
原始传统目录、链接内 VLA 代码、模型权重及已有结果均不修改。

## 2. 目标生成规则

新代码位于 [approach_geometry.py](/data1/xuhuixin/ula/adapters/approach_geometry.py)。
输入为 FoundationPose 估计的杯子世界位姿、模型几何元数据、所用手臂和
机器人运动学给出的 EE→TCP 变换。

模型元数据只使用 `center`、`extents`、`scale` 和可选 `transform_matrix`。
不访问抓取接触点、抓取描述、功能点或目标点来生成接近姿态。

1. 由中心、尺寸生成模型局部包围盒的 8 个角点。
2. 对角点应用模型缩放、模型变换以及视觉估计的世界位姿，得到世界坐标范围。
3. 取世界包围盒中心作为基础，左臂向世界 -X、右臂向世界 +X 偏移。
4. TCP 高度设为包围盒世界最高点，加上 `geometry-top-clearance`。
5. 使用固定竖直向下的接近姿态模板，再按配置尝试少量偏航候选。
6. 用机器人自身 EE/TCP 位姿计算参考点转换，得到规划器所需的 EE 目标。

位置规则可写成：

```text
c = 世界包围盒中心
半宽 = (x_max - x_min) / 2

TCP.x = c.x ± edge_fraction × 半宽     # 左臂负号，右臂正号
TCP.y = c.y
TCP.z = z_max + top_clearance
```

`edge_fraction=0` 表示在中心上方，`1` 表示在所选侧的边缘上方。
这里的“上方间距”从估计包围盒最高点量到 TCP，不是 EE 到杯子中心的距离，
也不是原标注模式从最终抓取 EE 目标退让的距离。

机器人中 EE 与夹爪中心 TCP 不是同一点，不能混用：

```text
T_EE←TCP = inverse(T_world←EE_current) × T_world←TCP_current
T_world←EE_goal = T_world←TCP_goal × inverse(T_EE←TCP)
```

当前机器人中两者位置相差 0.12 m；代码从机器人运动学计算该变换，
没有把 0.12 m 另写成接近算法中的固定世界高度。

## 3. 参数和先验

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--approach-mode` | `annotated` | 新模式需明确选择 `geometry` |
| `--geometry-top-clearance` | `0.10` m | TCP 高于估计包围盒顶面的距离，必须为有限正数 |
| `--geometry-edge-fraction` | `1.0` | 水平偏侧比例，范围 [0,1] |
| `--geometry-yaw-offsets` | `0 -0.15 0.15` rad | 顺序尝试的偏航偏移；可设为 `0` 固定朝向 |
| `--position-tolerance` | `0.02` m | 实际 EE 相对新目标的位置到位阈值，不变 |
| `--orientation-tolerance` | `15`° | 实际 EE 相对新目标的姿态到位阈值，不变 |

`--pregrasp-distance` 只用于 `annotated`。几何模式若传入非默认的此参数会拒绝运行，
以免误以为它会改变新模式间距，应改用 `--geometry-top-clearance`。
`geometry` 可用于 `hybrid / approach-only / grasp-only`，不会把纯传统模式也悄悄改为无标注抓取。
后续另增不读杯子模型的 `pointcloud` 路径，见
[点云接近说明](/data1/xuhuixin/ula/docs/pointcloud_approach.md)；两种几何来源不要混淆。

几何模式仍有明确先验：已知物体模型、水平桌面/世界 +Z 向上、左右臂工作区，
以及固定的向下接近姿态模板。它不根据任意相机视图自动寻找最佳抓法，不分析杯口、
把手或夹爪稳定接触，也不保证 VLA 从任意新目标都能成功。

## 4. 规划、执行和交接

对每个候选先调用现有 CuRobo 接口作只读规划查询。失败候选不执行，不将
`task.move` 锁定为失败；遇到无效起始状态立即停止搜索。选中首个可行候选后，
通过 `take_dense_action` 执行已有轨迹一次，避免查询成功后又重复规划。

一旦开始执行，不会因执行失败而自动换另一个候选继续运动。执行返回后仍使用
实际末端位姿检查 2 cm / 15° 到位条件；仅返回 True 而没有到位不能交给 VLA。
若所有候选失败，保存候选诊断并拒绝交接，不偷偷回退到标注模式。

保留原有规划碰撞模型；“目标高于包围盒”本身不等于完整夹爪无碰撞证明。
当前仿真执行接口也不等同于实机安全监控。

## 5. 运行方法

单独检查接近，不加载 VLA：

```bash
cd /data1/xuhuixin/ula
GPU=5 bash scripts/run_place_empty_cup.sh \
  --mode approach-only --approach-mode geometry \
  --episodes 1 --seed 0 \
  --output /data1/xuhuixin/ula/results/geometry_approach_only_seed0
```

完整三阶段实验，使用新输出目录：

```bash
cd /data1/xuhuixin/ula
GPU=5 bash scripts/run_place_empty_cup.sh \
  --mode hybrid --approach-mode geometry \
  --geometry-top-clearance 0.10 \
  --geometry-edge-fraction 1.0 \
  --geometry-yaw-offsets 0 -0.15 0.15 \
  --episodes 10 --seed 0 --max-vla-steps 500 \
  --output /data1/xuhuixin/ula/results/geometry_approach_10eps_repeat
```

原标注基线使用相同参数和另一输出目录，只改为 `--approach-mode annotated`。
请运行前检查 GPU 空闲情况；示例 GPU 编号不代表长期保留。

## 6. 审计与验证

报告顶部新增 `approach_mode` 和 `approach_configuration`。各轮 `handoff` 新增：

- `uses_contact_annotations=false`。
- `geometry_approach`：几何来源、世界包围盒、TCP/EE 转换、候选参数和规划结果。
- `selected_candidate`：选中哪个候选；无可行候选时为空并记录失败原因。
- `target_ee_pose` 和每个候选的 `target_tcp_pose`：分清两个参考点。
- `pregrasp_distance_m=null`：防止把原标注模式参数误当作新模式的间距。

计时新增 `geometry_target_generation` 子项，其余分段和排他计时口径不变。

已增加无 GPU 测试，覆盖：删除全部接触点字段后仍能生成目标、禁止访问接触字段、
包围盒旋转和缩放、模型变换、TCP/EE 换算、左右镜像、候选失败不执行、无效起点、
全部失败拒绝交接，以及执行返回成功但末端未到位仍拒绝交接。

另对旧实验 50 帧视觉估计作离线目标对比，生成器仅接收几何字段：新旧 EE 目标
位置差平均约 7.8 mm、最大约 9.5 mm，姿态差平均约 1.06°、最大约 4.96°。
**这只是离线目标对比，不是新控制成功率。** 它说明新规则从已有成功姿态附近起步，
不能证明接管范围已扩大或新物体泛化。

本次实际验证在 GPU 5 上完成 seeds 0–9：10/10 成功，两次交接均 10/10 通过，
排除首轮后 9 轮平均 6.978370 s。几何接近全部使用第一个候选；原标注模式的
`grasp_target_generation` 未被调用。42 项测试及运行期字段、计时加总检查通过。
完整结果见 [10 轮验证报告](/data1/xuhuixin/ula/results/geometry_approach_10eps_seed0_gpu5/SUMMARY.md)。
尚未对该模式运行 50 轮，也未验证任意新物体，不能以旧标注版 50/50 替代。

本次改造只移除接近阶段的抓取接触点依赖。FoundationPose 仍使用已知网格，
传统放置仍使用功能点元数据，抓稳交回仍依赖仿真真值。
