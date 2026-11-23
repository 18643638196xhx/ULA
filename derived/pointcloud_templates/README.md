# 50 任务单演示点云模板

每个任务一份 `template.npz`，由**单条示范的首帧**感知点云构成，是运行时感知的模板库：

- 物体点云（如 `hammer_points_world`、`shoe_points_world`）
- 支撑物点云（如 `block_points_world`、`pad_points_world`）

`UnifiedVisionPerception` / `estimate_template_z_yaw_rotation` 通过它做分割校验与 yaw 估计。

## 文件

| 文件 | 用途 |
|---|---|
| `template.npz` | 点云模板（必需）|
| `report.json` | 生成时的质量报告（点数、覆盖率、audit）|
| `offline_template_masks.npz` | 离线掩码，便于复核分割 |
| `rgbd.npz` | 生成模板用的那一帧 RGB-D |
| `template_audit.png` | 可视化审计图 |

## 来源

```
原路径: results/demo_clean_pointcloud_templates_50_20261120/tasks/<task>/
生成脚本: scripts/build_grasp_cloud_template.py
```

## 为什么入库

`results/` 被 `.gitignore` 忽略，而**所有新接入/升级的任务**（`pick_dual_bottles`、`pick_diverse_bottles`、`place_bread_basket`、`handover_mic`、`handover_block` 等）都依赖这批模板才能完成感知。合计 8.8 MB。
