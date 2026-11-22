# ULA 统一视觉定位模块：第一版

本文保留第一版设计和验证记录。当前代码已完成通用证据与分割恢复迁移，新增兼容的可选查询提示；最新实现、配置边界及同图比较见 [迁移报告](/data1/xuhuixin/ula/docs/unified_vision_evidence_migration_20261110.md)。第一版的代码快照与预测仍保留，没有覆盖。

## 本次交付范围

新增一个真实可调用的统一模块，所有目标使用同一套检测、分割、RGB-D 几何、候选选择、多视角核对和历史管理代码。运行逻辑不根据杯子、药瓶、微波炉等任务名称分支。目标名称仍由人工配置，不包含语言任务规划。

入口为 `UnifiedVision.observe(frames, queries, episode_id=...)`；策略环境中使用同签名的 `UnifiedVisionWorker.observe(...)`。后者在现有 Python 3.10 视觉环境中常驻 DINO/SAM，供 Python 3.11 策略调用，整批只加载一次模型。

没有替换原任务入口、修改控制器、模型权重或外部项目。旧任务仍使用原来的感知实现。本模块先作为独立感知能力搭建和验证，不将新模块的样例输出直接交给机器人。

## 公共处理流程

1. 根据查询中的文字提示调用同一个 GroundingDINO，SAM 产生候选掩码；不是最高分即认定正确。
2. 使用显式米/毫米单位、相机内参和世界到相机外参，把有效深度转换到统一世界坐标。
3. 对候选计算可见范围、可见表面中心，以及请求的平面或主轴。若存在满足必需几何属性的候选，不让一个分数更高但几何不合适的候选将其挤掉。
4. 对“主体—局部结构”查询，先定位主体，再在主体邻域内搜索相对其测得表面向外突出的结构；如要求特定名称，还需要相应的语义区域支持。没有语义支持的突出结构不能直接命名为所需把手。
5. 核对当前多相机观测。可见表面中心相差过大时，不因某个检测分数高就强行融合；历史预测也不能抹掉当前相机间的矛盾。
6. 记录本轮测得表面。下一轮需要运动估计时，使用通用三维局部 ICP，并检查重叠、残差和表面信息秩。平面或对称表面可能无法约束完整运动，不能仅凭低残差认定六自由度运动可靠。

每个目标、每个视角最多 4 个语义候选，最多两种分割提示（框、框加正点），每批最多 3 个视角。调用方提供已经采集的同步 RGB-D，不额外移动机器人。第一版会处理传入的所有视角，不是主动控制相机取景。

## 主体与突出结构

这是一种通用几何搜索，不是专门的“门把手检测器”。主体有明显主表面时使用受支持的平面；存在凹陷或多层表面时提取多个三维表面片；曲面上则使用局部切平面。没有固定竖直门板、世界 Z 方向长条或特定物体类别的判断。

突出是相对于主体的局部表面判断，而不是简单选离相机更近的点。候选还需有测得表面支持、空间邻近及连通性。主体关联仍只是几何证据，不证明物理连接；多个候选有歧义时保留诊断并拒绝指定其中一个。

`structures` 可以输出未命名结构；`parent` 查询则用于搜索特定或未命名的局部部位。例如篮子配置可以搜索未命名突出结构，不必预先叫它“门把手”。所有结构都明确标注 `graspability_evaluated=false` 和 `graspable=null`，没有宣称它们可以被夹爪安全抓取。

默认突出高度范围 12–120 mm、局部表面邻域 45 mm、邻接距离 30 mm 等均集中在 `VisionConfig`，是工程搜索范围，不是所有物体都适用的物理规律。曲面近似、凹腔、遮挡和邻近杂物仍可能引起误检或拒绝。

## 查询与输出

目标查询包含：

- `key`：本轮目标标识，不能在同一 episode 中改名后复用旧目标轨迹。
- `prompts`：人工指定的检测短语；未命名结构查询可以为空。
- `attributes`：请求 `position / bounds / plane / axis / structures / motion` 中的哪些属性。
- `required`：本次查询至少需要哪些输出，**不等于机器人可以执行或释放**。
- `parent`、`search_protrusions`：可选主体关系和结构搜索要求。
- `min_extent_m / max_extent_m`：可选、明确给出的世界轴向可见范围先验。

输出统一为 `TargetObservation`，包含状态、各属性可用性、当前测量、候选、拒绝原因、相机证据及非当前的历史记录。

必须注意：

- `position_world` 是可见点云范围中心，不是隐藏物体中心或杯底。
- `bounds_world` 不代表观察到了物体背面或完整尺寸。
- `axis` 是可见点的无向主轴，不一定是物体的语义朝向。
- `plane.visible_projection_hull_world` 是观测投影的外包边界，不证明其内部无孔洞、无障碍或适合放置。
- `full_object_pose_world` 当前为 null；没有偷偷补一个单位旋转冒充完整位姿。
- `position_error_bound_m` 当前为 null；不捏造已校准的毫米级误差保证。平面残差、运动信息秩等质量指标单独保存。
- `partial` 表示部分属性缺失；是否满足查询最低要求查看 `required_available`，实际操作许可仍由控制器判断。
- 遮挡或拒绝时，只可返回带 `usable_for_current_control=false` 的 `last_known`；不能将旧点云当作本轮新观测。

## 文件

| 文件 | 作用 |
| --- | --- |
| `adapters/unified_vision/contracts.py` | 传感器、查询、结果和共享配置 |
| `adapters/unified_vision/backends.py` | 共用现有本地 DINO/SAM 模型加载和任意文本查询 |
| `adapters/unified_vision/geometry.py` | 坐标转换、可见几何、多层/曲面表面、突出候选 |
| `adapters/unified_vision/engine.py` | 唯一主流程、父子查询、多视角与候选选择 |
| `adapters/unified_vision/tracking.py` | 通用测得表面局部运动估计与可观测性检查 |
| `adapters/unified_vision/worker.py` | 策略环境到常驻视觉环境的客户端 |
| `configs/unified_vision/*.json` | 杯子/垫子、药瓶/垫子、主体/局部结构、篮子结构的示例查询 |
| `scripts/run_unified_vision.py/.sh` | 单次回放、批量 JSONL 回放、常驻服务 |
| `scripts/smoke_unified_vision_worker.py` | 使用同一进程回放三个已有传感样例 |
| `tests/test_unified_vision.py` | 统一视觉专项测试 |

这些示例配置只更换目标名称和请求属性，不选择不同的杯子/药瓶/微波炉视觉类。篮子文件目前是接口示例，不代表已有真实篮子评测结果。

## 使用

单次传感文件回放，输出目录必须是新的：

```bash
cd /data1/xuhuixin/ula
GPU=0 bash scripts/run_unified_vision.sh \
  --config configs/unified_vision/cup_pad.json \
  --frame results/cup_initial_randomized_50scenes_seed0_20261101_v2/episode_000_seed0/rgbd.npz \
  --camera head --depth-unit mm \
  --output /data1/xuhuixin/ula/results/unified_vision_cup_new_run
```

策略程序中的常驻调用：

```python
from adapters.unified_vision import RGBDFrame, TargetQuery
from adapters.unified_vision.worker import UnifiedVisionWorker

# camera 是调用方当前采集的 RGB-D 字典，不是物体真值。
frame = RGBDFrame(
    camera="head", capture_id="capture_001",
    rgb=camera["rgb"][..., :3], depth=camera["depth"],
    intrinsic=camera["intrinsic_cv"], world_to_camera=camera["extrinsic_cv"],
    depth_unit="mm", timestamp_s=None,
)
queries = [TargetQuery("object", ("pill bottle",)),
           TargetQuery("destination", ("blue square pad",),
                       attributes=("position", "bounds", "plane"),
                       required=("position", "plane"))]

# 实际批量使用时，将 with 放在整批外部，跨轮次复用 worker。
with UnifiedVisionWorker("/data1/xuhuixin/ula",
                        "/data1/xuhuixin/ula/results/unified_worker_new_run",
                        gpu=0) as vision:
    observation = vision.observe([frame], queries, episode_id="episode_000")
```

多个视角必须拥有同一 `capture_id`；有时间戳时必须全部提供并满足最大时间差。缺时间戳时明确记录“调用方声明同一采集组”，不伪造同步测量。重复或倒序采集不能作为新帧更新轨迹；切换 episode 自动清除上一轮历史。

每个结果保存 `prediction.json`、所选表面点云和各视角分割证据 NPZ。内含 `box_xyxy`、相机标定、采集标识、候选来源和缺失属性，方便核对，而不是只保存一个坐标。

## 验证与边界

全量测试和实际回放结果记录在本文件末尾，未完成前不预填成功率。真实回放只读 RGB、深度和标定；输入加载器对白名单之外的数组不读取。没有使用 actor 位姿、身份像素、接触信息或任务评分，也没有构造仿真场景或执行机器人动作。

本版不是把旧任务全部替换后重新测过成功率，也不是已将历史优化全部迁移完成。当前尚未纳入旧模块的任务颜色启发式、基于桌面关系的目标恢复、CAD/FoundationPose 位姿后端、物体形状补全和完整不确定性标定。旧模块继续保留。当前通用主轴和点云中心也不能直接替代旧控制器要求的完整物体位姿与功能锚点。

下一步应冻结该接口，先做已有 50 场传感数据的身份、位置、拒绝原因专项评测，再决定替换哪些旧感知调用；不是立即启动全任务或宣称已泛化所有物体。

该离线评测已于 2026-11-10 完成，详见 [冻结评测报告](/data1/xuhuixin/ula/docs/unified_vision_archives_20261110.md)。实际使用杯子 42 帧、药瓶 50 帧和微波炉 50 帧；没有补生成场景。杯子与杯垫同时定位通过 30/42，药瓶与垫子 9/50，微波炉把手有输出 15/50 但缺少独立标签，正确率未知。评测过程中没有修改本节描述的接口和实现，当前还不能按旧感知模块的表现直接替换。

### 本轮实际验证结果

新增 **32 项专项测试**，与原项目测试合并后 **560 项通过**。日志为 `results/unified_vision_tests_20261110_v3.log`。测试覆盖显式深度单位与坐标转换、标定错误拒绝、传感器数据白名单、不可变帧快照、任意目标名称、语义高分与几何冲突、多个候选歧义、多视角不一致、重复/倒序帧、历史过期、主体/局部结构、平面/多层/曲面突出、局部运动退化、客户端协议和进程复用。

实际使用本地 DINO/SAM 模型，回放三个既有随机场景的初始 RGB-D：

| 样例 | 查询最低要求 | 当前输出情况 |
| --- | --- | --- |
| 杯子 seed 0 | 物体位置与范围；杯垫位置与平面 | 两个查询均满足；杯子主轴不可靠，保留缺失而不强行补齐 |
| 药瓶 seed 0 | 物体位置与范围；蓝垫位置与平面 | 两个查询均满足；另输出了可见表面的主轴 |
| 门板—局部结构 seed 0 | 主体位置；有语义支持的突出局部位置 | 两个查询均满足；主体整体平面不可靠，但多层表面搜索支持局部候选 |

正式接口冒烟报告：`results/unified_vision_worker_smoke_20261110_v4/report.json`。同一 PID 1754591 连续完成 3 个请求，`starts=1`，`completed_requests=3`，`sources_unchanged=true`，最终 `closed=true`、`running=false`。全部六个查询满足的是配置要求，不是经独立身份/位置核验得到的“6/6 定位正确”，更不是任务成功率。没有计算毫米误差或使用任务真值给出精度结论。

曲面和多个突出结构已用合成三维数据验证通用算子；尚未对真实篮子场景做回放或 50 场评测。所选掩码、表面与标定证据已保存于 `worker/request_*/prediction/`，可用于后续逐例核验。

早期开发记录保留：初次在策略 Python 环境加载视觉模型因缺少 `yapf` 失败，没有安装依赖，而是复用现有专用视觉环境；第一轮语义候选没有按必需平面属性优先筛选，杯垫出现歧义；单一主表面/局部表面未覆盖凹陷多层结构，把手没有输出。后两项通过通用几何属性筛选和三维多层表面提取修正，没有加入 seed 编号或任务名称特例。
