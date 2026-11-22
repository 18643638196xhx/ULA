# 三类定位问题的通用修改

日期：2026-11-12。修改位置输出、候选身份和重复候选/跨视角筛选；不更改任务控制、CAD 接受数值门槛或旧实验结果。

## 使用方式与默认兼容

新候选身份及可见面恢复策略由显式配置启用：

[robust_geometry.json](/data1/xuhuixin/ula/configs/unified_vision/robust_geometry.json)

其中 `require_semantic_identity=true`、`complete_visible_surface=true`。在已有回放入口使用 `--geometry-config` 传入；调用 Python 接口时传入同名 `VisionConfig` 字段。默认值为 false，已有控制器不会自动切换新策略。是否采用多视角仍是独立的 `require_cross_view_cad` 配置，并不因开启这份配置而自动增加相机。

## 1. 位置输出不再混用

`TargetQuery.position_reference` 默认 `visible_surface`，保留旧的 `position_world` 含义。有 CAD 时额外输出 `visible_surface_position_world` 和 `cad_origin_world`。

调用方可以明确选：

- `cad_origin`：输出已核验模型坐标原点。
- `cad_anchor`：同时明确提供静态 CAD 坐标系下的 `anchor_object_m`，将其随模型位姿变换到世界坐标。

锚点不自动代表杯底或桶口，也不会从仿真接触点、功能点读取。无 CAD 不接受这两种位置请求；未提供锚点不接受 `cad_anchor`。模型推断位置通过 `position_is_model_inferred` 标注；隐藏部分仍不声称被直接测到，完整位姿的对称性也仍不声称已全局解决。

原可见中心评分显式选用 `visible_surface_position_world`，不将 CAD 原点或锚点评为可见中心。没有可见中心时不能借模型坐标通过旧评分。

## 2. 候选身份证据

每个分割候选保存文字检测提示、检测分数和掩码落在对应文字框中的比例。检测分数不是身份概率，检测框支持也不是身份真值。

严格模式要求候选具备空间重合的文字检测支持，才允许其 CAD 核验结果进入最终选择。无语义来源的纯 RGB-D 区域仍可生成和检查，但不能仅因 CAD 局部拟合通过就直接叫作目标。显式外观先验仍按原逻辑检查；本次没有增加新外观网络，也没有实现 CAD 纹理身份识别。

为避免“检测器没交出高分框就直接拒绝”，严格模式的 CAD 恢复轮允许将检测候选阈值暂时降到通用的 0.15（`recovery_detection_threshold`），只扩大候选搜索，不改变 CAD 表面/渲染接受门槛。首轮阈值不变；恢复调用后即恢复共享检测器阈值，异常时也恢复。缓存区分首轮和恢复阈值，避免错误复用高阈值的空结果。这仍不是身份概率，也不保证低分候选一定正确。

若几何通过但身份证据不足，输出 `no_semantically_supported_CAD_candidate`，与 `no_CAD_consistent_current_candidate` 分开。这可能减少错误输出，也可能增加拒绝，收益必须用独立标签统计，不能只看输出率。

## 3. 已核验模型辅助恢复可见区域

只在候选 CAD 已通过现有检查后，利用该位姿的渲染轮廓寻找遗漏像素。新增像素必须同时具备实际有效深度，并与渲染深度相差不超过原 8 mm 中位数误差限；不生成被遮挡的深度，不把 CAD 隐藏面当作测量。

新增像素并入原候选后，重新检查原渲染门槛和精确三角面距离门槛，通过才重建掩码、点云和可见位置。记录 `visible_completion` 审计及新增像素数。原来参与选择的几何成本不因补点自动获得额外奖励。

这不是“所有缺失形状都能补回”：CAD 位姿本身不准或遗漏部位没有有效深度时仍无法恢复。

## 4. 歧义与跨视角衔接

同一视角里两个不同的局部分割，若核验后的 CAD 原点接近、渲染轮廓重合且渲染深度一致，可识别为同一实例的重复候选，避免把同一物体的多个片段当成两个不同目标。仍保留全部原始候选诊断记录；不同物体不能仅凭分数接近就合并。

显式开启同位姿跨视角模式时，不再额外用旧的“各视角独立估计的中心必须接近”规则直接拒绝全部候选。真正的跨视角深度矛盾检查、候选成本差门槛和联合一对一分配仍保留。模型自由旋转对称性并未由此自动解决。

## 验证

首版全量回归 671 项通过；加入恢复检测后的最终测试见下方日志。用例覆盖位置语义、静态锚点旋转、旧指标保护、严格身份、可见像素恢复、无效/遮挡深度不生成点、重复 CAD 轮廓去重及异常恢复检测阈值。

- [全量测试日志](/data1/xuhuixin/ula/results/unified_vision_robust_geometry_tests_20261112_v2.log)
- [新增测试](/data1/xuhuixin/ula/tests/test_unified_vision_robust_geometry.py)

已启动 8 场归档头部 RGB-D 的真实 DINO/SAM/FoundationPose 回放：摇瓶、平底锅、订书机加垫子、微波炉各取前两个已完成的有效场景，不按成功或失败挑选。输入是旧归档，并非重新建立的场景；也不增加相机。

- [最终小样本回放报告](/data1/xuhuixin/ula/results/unified_vision_robust_geometry_replay8_20261112_v2/report.json)
- [首轮退步结果（保留）](/data1/xuhuixin/ula/results/unified_vision_robust_geometry_replay8_20261112_v1/report.json)
- [可重复回放入口](/data1/xuhuixin/ula/scripts/replay_unified_vision_robust.py)

预测保存锁定后才读取本轮独立标签，评分不回传算法。小样本不代表新的 50 场成功率，更不是完整机器人任务成功率。没有重跑或覆盖 2500 场历史评测。

首轮八场的原归档通过数为 7，新策略通过 3，退步包含订书机和微波炉因缺少语义支持而拒绝。因此保持为显式实验配置，不默认替换。另需注意原归档距离核验仍是旧采样版，所以这不是只改变本轮三项功能的严格单因素消融。

本轮没有实现新训练模型、CAD 纹理识别或联合多视角位姿优化；未声称歧义和不完整形状问题已全部解决。
