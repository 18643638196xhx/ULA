# 已知模型药瓶：50 轮冻结参数评测

本次评测使用此前 v12 完整成功 4/5 的方案：传统视觉接近 → VLA 闭爪接近候选 → 传统视觉定位、搬运、下降、松爪、撤离。不是 VLA 连续放置对照，也不是无模型点云方案。

固定型号 `080_pillbottle/base1`，接近为 `annotated`、接触区域为 `all`，不启用试抬。位置使用官方任务随机生成。采用连续 seeds **5–54**，避开参与调试的 0–4，不筛种子、不按失败重跑替换结果。运行期间不修改控制代码或参数。既有开发样本 4/5 不合并进本批。

## 启动命令

```bash
cd /data1/xuhuixin/ula
nohup env GPU=7 bash scripts/run_move_pillbottle_pad_known_model.sh \
  --known-model 1 --known-approach annotated --known-contact-region all \
  --episodes 50 --seed 5 \
  --output /data1/xuhuixin/ula/results/pillbottle_known_traditional_finish_50eps_20261031_seed5 \
  > /data1/xuhuixin/ula/results/pillbottle_known_traditional_finish_50eps_20261031_seed5.log \
  2>&1 < /dev/null &
```

共享 GPU 7，不中止已有其他任务；使用原有分配器预算，不代表硬显存上限或算力隔离。

## 统计口径

完整流程成功要求控制流程走完且官方最终评分成功；单列官方最终评分成功，不能把 VLA 自行放好但没有完成传统接管的轮次算成完整 ULA 成功。成功率分母为全部 50 轮，首轮也计入。

耗时分别列出全部轮次和剔除首轮 seed 5 的结果；“平均成功完成耗时”只在完整成功轮次上计算，包含建立场景到最终评测/保存截图的单轮墙钟时间，首轮还包含延迟感知加载。失败尝试耗时不能混称为完成时间。另保存传统接近、VLA、传统放置等分段计时。共享 GPU 下的墙钟耗时不等同于实机物理执行时长。

控制不读取实时物体/垫子仿真位姿、接触信息或在线官方成功信号；控制结束后才用真值评分。明确使用已知 CAD/标注、RGB-D、机器人模型与状态、相机标定及配置的桌面碰撞盒，不是无先验方案。

批次 `report.json` 每结束一轮更新一次；`successes` 为完整成功数，`task_successes` 为官方成功数。已知模型专用的顶层说明和工作进程退出/保护文件校验在批次结束时补齐；运行中不要仅根据继承的顶层旧说明判断是否执行过试抬，应看配置 `trial_lift_enabled=false` 与每轮跳过验证记录。

## 进度

```bash
tail -f /data1/xuhuixin/ula/results/pillbottle_known_traditional_finish_50eps_20261031_seed5.log
```

最终以全部 50 个每轮结果、工作进程关闭和保护文件指纹一致为批次完成依据。个别轮次失败会使程序最终返回非零状态，不等于整批未运行完；需要核对 `completed_episodes` 和 `run_status`。
