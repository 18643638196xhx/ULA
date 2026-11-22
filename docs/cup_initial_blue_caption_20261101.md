# 初始识别对照：明确提示蓝色杯子

## 用户要求与范围

在上一批仅抓取前识别测试中，用“蓝色杯子”重新识别。不执行接近、抓取、试抬、放置，不运行 VLA，不计算任务成功率。

为只比较提示词，直接重放上批 `cup_initial_randomized_50scenes_seed0_20261101_v2` 保存的原始 RGB-D，而非重新随机生成场景。原 50 个种子中，42 个有有效图像，会全部重新推理；原初始化失败的 4、6、12、16、26、31、35、38 没有图像，本次标记 `source_unavailable`，没有重新初始化或补种子，不能声称对这 8 个完成了识别测试。

## 唯一识别改动

杯子的 DINO caption 从 `cup . mug .` 换成 `blue cup .`，不回退到通用 cup/mug 提示。杯垫仍使用原提示和算法；SAM、几何检查、软颜色排序、候选歧义门槛和掩码核验标准保持不变。

**“蓝色杯子”的语义提示，不等于固定蓝色像素硬筛选。** 颜色仍最多给排序增加 0.10，不要求每个候选必须包含指定比例蓝色像素，避免把偏色直接等同于不是杯子。此次不是恢复旧 `blue_cup_candidates` 硬颜色候选生成。

通过独立类、worker 和重放入口实现，旧初始识别入口及原控制器不改动。新类审计明确记录实际 caption 和无通用提示回退。

## 输入与独立核验

每张新输入都是上批 `rgbd.npz` 的逐字节复制，保存 SHA256 并校验一致。视觉只接收 RGB、深度、相机内外参四项。新的预测写出冻结后，才打开同一原帧保存的目标实例标签核验；标签不参与选择、推理或重试。

依旧使用可见目标掩码 IoU≥0.5 判断匹配，不将“有输出”当成认对，不推断抓取/放置成功。每帧另外比较新旧杯垫预测是否逐像素相同，核实杯垫分支未因本次杯子提示而变化。

3 项新 CPU 回归验证提示词替换、杯垫提示保留、0 蓝色像素不被硬拒绝以及独立入口加载；全量 **300 项通过**。

## 运行

```bash
cd /data1/xuhuixin/ula
GPU=7 ULA_CUP_RUNNER=/data1/xuhuixin/ula/scripts/replay_cup_initial_blue.py \
MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash scripts/run_place_empty_cup_randomized.sh \
  --input /data1/xuhuixin/ula/results/cup_initial_randomized_50scenes_seed0_20261101_v2 \
  --output /data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1
```

GPU7 共享，仅视觉模型常驻，视觉 PyTorch 分配器预算 10%，不是硬显存或算力隔离。本次不启动仿真或运动规划器。

完整日志为输出同名前缀 `.launcher.log`；新输出保存各帧原图、预测、核验掩码和标注图，不修改旧结果。青色/粉色为杯子/杯垫预测，黄色轮廓仅为事后核验目标。总览 `contact_sheet_01.png` 至 `06.png` 覆盖原 50 个种子。

## 最终结果：同图重放已结束

原 50 个种子均已记录，42 张有效图像全部完成新推理和核验；8 个原无图种子没有重新生成或伪造识别结果。新视觉进程启动一次、42 次请求全部返回、已正常关闭。没有启动仿真、运动规划或任务动作。

| 42 张完全相同的有效画面 | 通用 cup/mug 提示 | blue cup 提示 |
| --- | --- | --- |
| 杯子有输出 | 33 | **42** |
| 杯子匹配，掩码 IoU≥0.5 | 30 | **40** |
| 杯子认错 | 3 | **2** |
| 杯子未输出 | 9 | **0** |
| 杯垫匹配 | 40 | **40** |
| 两者同时匹配 | 28 | **38** |

杯垫仍为 41 个有输出、40 个匹配；seed 17 选错盘状物，19 无输出。所有 42 张新旧杯垫预测逐像素完全相同，核实了本次杯垫结果未变化。

杯子新增匹配 seeds：**5、10、14、28、36、40、43、46、48、49**，没有从匹配退化到不匹配的种子。旧认错的 14、46 已匹配；旧认错的 1 仍认错。旧无输出的 11 现在有输出但认错，不能把这一变化当作改善。

### 剩余两次杯子认错

- **seed 1**：仍选中右上方黑色圆筒容器，目标杯子在左下边缘。预测 3454 像素，与目标 1466 个可见像素的交集为 0。
- **seed 11**：选中右侧蓝色盒状物，目标杯子在左侧边缘。预测 1847 像素，与目标 1498 个可见像素的交集为 0。

两张目标都靠近左边界且部分被机械臂遮挡，这是可见的困难条件；尚未做遮挡/边界单因素实验，不能仅凭截图认定唯一原因。蓝色语义并不保证选中物体就是杯子，蓝色杂物仍可通过目前的粗几何。

为避免把原图中蓝杯本来的颜色误当成预测，以下另用红框表示模型选择、绿框表示目标杯子：

- [seed 1 红绿框核对](/data1/xuhuixin/ula/results/cup_initial_blue_selection_clarification_20261101_v1/seed_01_selection.png)
- [seed 11 红绿框核对](/data1/xuhuixin/ula/results/cup_initial_blue_selection_clarification_20261101_v1/seed_11_selection.png)

### 全部画面

- [第 1 页：seed 0–8](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_01.png)
- [第 2 页：seed 9–17](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_02.png)
- [第 3 页：seed 18–26](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_03.png)
- [第 4 页：seed 27–35](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_04.png)
- [第 5 页：seed 36–44](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_05.png)
- [第 6 页：seed 45–49](/data1/xuhuixin/ula/results/cup_initial_blue_caption_replay50_seed0_20261101_v1/contact_sheet_06.png)

### 一致性检查与结论

42 个新输入均与旧输入 SHA256 一致；新的预测 JSON 均先于本次核验掩码文件保存；所有有效帧记录 `cup_semantic_caption=blue cup .`、`color_is_hard_gate=false`。来源哈希前后一致，旧入口与旧输出未改动。排除第一张模型加载后，41 张平均感知墙钟时间约 0.399 s，仅供同机共享条件下参考，不是任务完成时间。

此次明确的蓝杯语义使初始识别明显改善，但仅验证了保存初始画面的识别，不验证抓后遮挡跟踪、接近、抓取或放置。新分支仍为独立实验，没有据此宣布整体任务成功率。
