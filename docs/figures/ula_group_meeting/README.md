# ULA 组会图表

所有架构/原理图采用浅蓝圆角框、蓝色文字、灰色箭头；浅橙色专门提示控制切换中的仿真真值依赖。

- `01_control_layers`：传统控制的三层职责。
- `02_architecture`：ULA 调度、两条控制分支、共用执行与反馈。
- `03_episode_sequence`：最新 pointcloud 批次 seed 0 的四阶段真实截图。
- `04_known_model`：已知模型 + FoundationPose + 元数据路线。
- `05_pointcloud_approach`：点云可见范围与 TCP 接近目标。
- `06_pointcloud_placement`：在线参考、前后 ICP 与近似支撑锚点。
- `07_handoff`：两次交接条件。
- `08_results`：由原始 JSON 自动计算的成功轮分段均值。
- `09_iterations`：90% → 92% → 96% 迭代与剩余失败。
- `10_truth_boundary`：视觉、机器人信息、控制真值与评测真值的区别。

每个图有 PNG（可直接插入报告）和 SVG（矢量、文字可编辑）。SVG 使用文泉驿微米黑；其他电脑没有该字体时可能发生替代，PNG 和 PDF 不依赖本机字体。

`page_01.png` 至 `page_14.png` 是投屏版页面预览；`page_contact_sheet.png` 为总览。
`report_sources.json` 记录数据来源、SHA256 与均值样本数。

重建脚本：`/data1/xuhuixin/ula/scripts/build_group_meeting_report.py`。只读原实验结果，不启动机器人实验，不修改控制代码或原始 tra_pipe 目录。
