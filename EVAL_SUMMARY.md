# ULA 评测结果汇总

`demo_randomized` 随机域（随机背景、桌面杂物、台面高度 ±3cm、随机光照）。

| 任务 | 局数 | 成功 | 原始成功率 | 剔除场景异常 |
|---|---:|---:|---:|---:|
| `handover_mic (shard0)` | 25 | 25 | 100.0% | 100.0% |
| `place_empty_cup` | 50 | 49 | 98.0% | 98.0% |
| `click_bell` | 50 | 49 | 98.0% | 98.0% |
| `grab_roller` | 50 | 46 | 92.0% | 92.0% |
| `handover_mic (shard1)` | 25 | 23 | 92.0% | 92.0% |
| `click_alarmclock` | 50 | 44 | 88.0% | 88.0% |
| `move_can_pot (no yaw)` | 50 | 41 | 82.0% | 87.2% |
| `pick_dual_bottles` | 50 | 39 | 78.0% | 78.0% |
| `place_mouse_pad` | 50 | 38 | 76.0% | 76.0% |
| `place_container_plate` | 50 | 37 | 74.0% | 75.5% |
| `move_playingcard_away` | 50 | 36 | 72.0% | 87.8% |
| `place_shoe` | 50 | 31 | 62.0% | 62.0% |
| `move_stapler_pad` | 50 | 31 | 62.0% | 62.0% |
| `move_pillbottle_pad` | 50 | 28 | 56.0% | 56.0% |
| `adjust_bottle` | 50 | 25 | 50.0% | 65.8% |
| `move_can_pot (with yaw)` | 50 | 24 | 48.0% | 55.8% |
| `lift_pot` | 37 | 13 | 35.1% | 35.1% |
| `handover_block` | 5 | 0 | 0.0% | 0.0% |

## 说明

- 「剔除场景异常」= 去掉 `scene_or_runtime_error`（OOM、物体未站稳等环境侧失败）后的成功率。
- `shot` 目录下每局保留精简摘要（seed / 成败 / 失败阶段 / 末态评估 / 各阶段结果）。
- 规则快照见 `rule_snapshots/<task>/<arm>.yaml`。
