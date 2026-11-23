# 派生数据（从示范中提取，体积小、可复现）

| 目录 | 内容 | 来源 |
|---|---|---|
| `tracks/` | 视觉轨迹 `tracks.npz`（每帧物体/目标位姿），规则就是从这里提取的 | `/data2/xuhuixin/ula_demo_rgbd` 的示范 HDF5 首帧感知 + 刚体传播 |
| `templates/` | 多视角点云模板 `merged_template.npz`（yaw 估计用，合并 6 个视角） | 同上，每条示范首帧感知点云按腕部 yaw 归一后合并 |
| `perception/` | 建模时那一帧的统一视觉估计 `estimate.npz`（点云 + 支撑物） | 同上 |

## 为什么单独保存

原始示范 HDF5 每条约 500 MB（深度图 float64 占 97%），全库 90 GB，不适合入库。
这些派生文件合计不到 10 MB，足以：

- 重新编译规则（`scripts/extract_demo_primitives.py`）
- 复核感知结果
- 复现多视角模板与 yaw 估计

## 与规则的关系

`rule_snapshots/<task>/<arm>.yaml` 是最终产物（相对位姿关系），可直接执行；
`derived/tracks/` 是它的输入，改了提取算法可以从这里重算，不必重采数据。
