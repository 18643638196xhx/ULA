# ULA 示范相对接近位姿改造交接文档

## 1. 项目位置与目标

项目目录：

```text
/data1/xuhuixin/ula
```

ULA 当前针对支持面放置任务采用以下分层流程：

```text
统一 RGB-D 视觉定位
→ 传统控制移动到目标物体附近
→ VLA 完成局部对准和抓取
→ 视觉确认物体是否随手移动
→ 传统控制搬运、下降和释放
→ RoboTwin 官方成功判定
```

本次修改针对第一段传统接近。旧方法主要调用：

```text
adapters/approach_geometry.py::generate_bounds_candidates
```

它只根据当前物体点云包围框生成顶部接近位置，再枚举少量偏航或俯仰角，没有使用官方示范中实际采用的接近方向。这会造成某些场景虽然更换了很多手腕姿态，但 TCP 位置基本不变，最终所有候选都 IK 失败。

本次改造目标：

```text
示范相对接近位姿                  第一优先级
→ 示范位姿附近的视觉误差候选      第二优先级
→ generate_bounds_candidates      仅作为兜底
→ CuRobo按顺序选择第一条可行路径
```

## 2. 当前已实现的候选优先级

在线运行时的候选顺序为：

1. 示范迁移后的原始接近候选；
2. 围绕示范候选生成的 4 个视觉误差邻域候选；
3. 示范位置搭配最多 2 个已经过几何模板验证的安全腕姿；
4. `generate_bounds_candidates` 生成的几何候选；
5. 几何兜底除原机械臂侧候选外，还包含朝向当前 TCP 的表面候选和按当前物体尺度生成的外侧候选；
6. CuRobo 按上述顺序进行 IK、碰撞检查和轨迹规划；
7. 只执行第一条规划成功的轨迹，一旦开始物理执行便不再尝试后续候选。

误差邻域的半径不是任务固定超参数，而是当前 RGB-D 物体点云的中位最近邻间距：

```python
uncertainty_m = measured_resolution(object_points)
```

因此不同相机距离、不同物体尺寸和不同点云密度会得到不同的候选邻域。

## 3. 两种示范先验模式

实现位于：

```text
adapters/demo_relative_approach.py
```

### 3.1 `object_6d`

如果离线示范中已获得可靠的物体六维位姿，则保存：

```text
object_to_ee = inverse(demo_object_world) × demo_ee_world
```

在线阶段使用当前视觉物体位姿：

```text
current_ee_world = current_object_world × object_to_ee
```

这是真正的物体坐标系相对位姿迁移。只有当前视觉也提供可靠物体六维位姿时才允许使用。

### 3.2 `visual_bounds_top_anchor`

RoboTwin 官方 `demo_clean/*.hdf5` 只包含：

- 机械臂末端轨迹；
- 夹爪状态；
- RGB 图像；
- 相机内外参；
- 任务语言。

官方 HDF5 不包含深度和物体位姿，因此不能直接计算严格的 `object_to_ee`。

当前九个任务使用诚实的降级模式：

1. 从示范轨迹找到闭爪动作之前约 5–12 cm 的接近参考帧；
2. 记录“交互末端位置到接近末端位置”的方向和距离；
3. 在线阶段以当前 RGB-D 物体包围框顶部中心作为 TCP 语义的视觉锚点；
4. 将示范接近位移迁移到当前视觉锚点；
5. 腕部朝向使用当前机械臂和当前场景的通用安全几何模板，不照搬示范中的世界坐标绝对旋转；
6. 由 `TCP × inverse(EE_to_TCP)` 计算 EE 目标，机器人标定偏移只应用一次。

此模式只保留示范的接近方向和距离，不保留绝对腕姿，也不声称获得了完整物体坐标系位姿。此前把视觉锚点当成 EE 后再次叠加 12 cm TCP 偏移，是约 19 cm 到位误差的主要来源之一。

## 4. 左右手处理

如果示范使用左手，而当前物体位置更适合右手，只对示范位移的世界坐标 X 分量做镜像。绝对末端旋转不参与迁移：

```text
delta_right = diag(-1, 1, 1) × delta_left
```

位移的 X 分量也取反。

完整 `object_6d` 模式不会自动镜像任意物体坐标系，因为那可能产生错误语义。该模式要求为对应机械臂显式编译先验。

## 5. 修改文件

### 新增核心模块

```text
adapters/demo_relative_approach.py
```

主要接口：

```python
load_demo_approach_prior(path)
generate_demo_relative_candidates(...)
prepend_demo_candidates(...)
```

### 新增示范编译脚本

```text
scripts/compile_demo_approach_prior.py
```

它读取此前由 `extract_demo_rules_50.py` 生成的单任务规则 JSON，提取：

- 示范机械臂；
- 接近参考帧；
- 闭爪前交互帧；
- 接近位移；
- 末端旋转；
- 原始示范路径和数据来源。

### 修改几何接近模块

```text
adapters/approach_geometry.py
```

`execute_pointcloud_approach` 新增可选参数：

```python
measured_points=None
object_pose_world=None
demo_prior_path=None
```

有先验时调用 `prepend_demo_candidates`，无先验时保持旧行为。

### 修改药瓶接近模块

```text
adapters/pillbottle.py
```

`approach_bottle` 现在先生成原有几何、俯仰、偏航候选，再把示范候选放到最前面。原双臂可达性回退仍然保留。

### 修改统一视觉输出

```text
adapters/unified_perception.py
```

`SupportPlacementEstimate` 新增：

```python
object_pose_world: np.ndarray | None
```

如果 CAD 核验输出 `full_object_pose_world`，统一视觉会保留该结果并写入参考 NPZ。没有可靠 CAD 位姿时仍使用可见点云范围。

### 修改运行入口

```text
scripts/run_place_empty_cup.py
scripts/run_move_pillbottle_pad.py
scripts/run_ula_task.py
```

杯子和药瓶基础运行器新增参数：

```text
--demo-approach-prior <JSON路径>
```

统一入口会根据注册表自动加入该参数，正常运行不需要手工传入。

### 修改任务注册表

```text
adapters/ula_task_registry.py
```

九个任务都增加了：

```python
demo_approach_prior="rules/approach_priors/<task>.json"
```

## 6. 已生成先验的九个任务

先验目录：

```text
rules/approach_priors/
```

当前包含：

```text
place_empty_cup.json
move_pillbottle_pad.json
place_fan.json
place_mouse_pad.json
move_stapler_pad.json
place_object_scale.json
place_object_stand.json
place_shoe.json
place_container_plate.json
```

当前九份先验均为：

```text
mode = visual_bounds_top_anchor
```

## 7. 测试与离线回放结果

新增测试：

```text
tests/test_demo_relative_approach.py
```

覆盖内容：

- 示范原始候选必须排第一；
- 4 个视觉误差邻域必须紧随其后；
- 安全腕姿候选位于视觉邻域之后；
- 几何候选只能排在示范相关候选之后；
- 邻域大小必须等于视觉测得的点云间距；
- 左右手镜像正确；
- 完整 `object_6d` 变换组合正确；
- 非刚体矩阵或错误 schema 会被拒绝。

本轮接近相关回归测试结果：

```text
49 passed
```

使用之前失败的 `move_pillbottle_pad random seed 3` 保存点云进行离线回放：

```text
物体点云分辨率：2.6768 mm
修正 TCP/EE 语义后的候选顺序：
  1 × demo_exact
  4 × demo_visual_uncertainty_neighbour
  2 × demo_safe_orientation_variant
  geometry_bounds_fallback
```

该场景生成的第一示范 TCP 位置：

```text
[-0.12858, 0.24295, 0.92154]
```

第一个旧几何 TCP 候选位置：

```text
[-0.13052, 0.26438, 0.86895]
```

并且所有候选均满足 `EE × EE_to_TCP = TCP`（数值误差约 `1.11e-16`）。

针对旧失败样本的机器人复验：

```text
place_empty_cup seed 6：通过，位置误差 2.04 mm，姿态误差 0.30°
place_fan seed 7：通过，位置误差 1.83 mm，姿态误差 0.27°
move_pillbottle_pad seed 0：通过，位置误差 1.97 mm，姿态误差 0.31°
```

上述三场都由第一条示范方向候选直接成功。`move_pillbottle_pad seed 3` 在仅修正 TCP/EE 后仍因所有候选位置过于集中而 IK 失败，因此又加入了通用的、由当前物体尺寸和当前 TCP 方向推导的位置壳层。加入后该场通过：选择朝当前 TCP 方向、位于观测包围球半径外侧的位置候选，位置误差 3.28 mm、姿态误差 0.34°。这说明 A4 不能仅靠增加腕姿解决，必须同时搜索位置。

## 8. random 机器人评测命令

### 8.1 先复验药瓶 seed 3

seed 3 之前的主要失败是左右手接近候选全部 IK 失败，适合优先验证此次修改。

```bash
cd /data1/xuhuixin/ula

OUT=/data1/xuhuixin/ula/results/move_pillbottle_pad_random_seed3_demo_approach_v2
LOG="${OUT}.launcher.log"
screen -dmS ula_pb_seed3 bash -lc "cd /data1/xuhuixin/ula && env GPU=1 ULA_TASK_CONFIG=demo_randomized bash scripts/run_ula_task.sh --task move_pillbottle_pad --mode hybrid --episodes 1 --seed 3 --instruction-from-demo --output ${OUT} > ${LOG} 2>&1"
```

查看进度：

```bash
tail -f /data1/xuhuixin/ula/results/move_pillbottle_pad_random_seed3_demo_approach_v2.launcher.log
```

### 8.2 药瓶 random 50 次完整评测

```bash
cd /data1/xuhuixin/ula

OUT=/data1/xuhuixin/ula/results/move_pillbottle_pad_random50_demo_approach_v2
LOG="${OUT}.launcher.log"
screen -dmS ula_pb_random50 bash -lc "cd /data1/xuhuixin/ula && env GPU=1 ULA_TASK_CONFIG=demo_randomized bash scripts/run_ula_task.sh --task move_pillbottle_pad --mode hybrid --episodes 50 --seed 0 --instruction-from-demo --output ${OUT} > ${LOG} 2>&1"
```

### 8.3 杯子 random 50 次完整评测

```bash
cd /data1/xuhuixin/ula

OUT=/data1/xuhuixin/ula/results/place_empty_cup_random50_demo_approach_v2
LOG="${OUT}.launcher.log"
screen -dmS ula_cup_random50 bash -lc "cd /data1/xuhuixin/ula && env GPU=2 ULA_TASK_CONFIG=demo_randomized bash scripts/run_ula_task.sh --task place_empty_cup --mode hybrid --episodes 50 --seed 0 --instruction-from-demo --output ${OUT} > ${LOG} 2>&1"
```

### 8.4 查看完成数量和进程

```bash
find results/move_pillbottle_pad_random50_demo_approach_v1 \
  -name result.json | wc -l

ps -ef | grep '[r]un_ula_task.py'
```

统一入口会自动从注册表加载示范先验，不需要额外指定 `--demo-approach-prior`。

## 9. 评测时重点检查的审计字段

每轮结果中的接近审计应包含：

```text
geometry_approach.source
geometry_approach.candidate_order
geometry_approach.demo_candidate_count
geometry_approach.geometry_fallback_count
geometry_approach.demonstration.prior_mode
geometry_approach.demonstration.neighbourhood_radius_m
geometry_approach.planning_attempts
geometry_approach.selected_candidate
geometry_approach.arm_reachability_attempts
```

预期：

```text
candidate_order =
  demo_exact
  demo_visual_uncertainty_neighbours
  demo_safe_orientation_variants
  geometry_bounds_fallback
```

需要统计：

1. 多少场直接选择 `demo_exact`；
2. 多少场选择视觉邻域候选；
3. 多少场最终回退到几何候选；
4. 多少场所有候选仍然 IK 失败；
5. 示范候选是否降低平均规划尝试数和传统接近时间；
6. 接近成功后 VLA 抓取成功率是否改善。

## 10. 当前边界和后续工作

1. 当前九个先验来自一条官方 clean 示范，只表达一种主要接近方式。
2. 官方示范缺少物体位姿，因此当前不是严格的六维物体相对变换。
3. 如果后续通过离线 CAD/RGB 拟合获得示范物体六维位姿，应将对应 JSON 升级为 `object_6d`。
4. 不要直接回放示范世界坐标或关节轨迹；random 场景必须重新视觉定位并由 CuRobo 规划。
5. 不要删除几何兜底，它用于处理示范候选不可达或示范外形不适配的场景。
6. 不要因为示范候选规划成功就认为抓取成功；VLA 抓取和后续视觉附着确认仍然必须保留。
7. 下一步应先运行 seed 3 单场，再运行药瓶 5 场，确认审计正确后才启动 50 场。

