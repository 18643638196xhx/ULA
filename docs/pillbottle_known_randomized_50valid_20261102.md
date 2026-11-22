# move_pillbottle_pad：最新已知模型 ULA 的随机场景测试

## 本次使用的方案

沿用药瓶最新 v24 的执行逻辑：视觉定位、传统接近、VLA 抓取、传统搬运放置。启用已知 CAD 标注接近、ranked 接近搜索、visual 放置、头部定位不可信后腕部回退，以及初始蓝色垫子高度校验和 RGB-D 恢复。不增加试抬，不放宽到位、配准或释放条件。

**这是已知 base1 药瓶模型的评测，不是无模型方案，也不是官方五种药瓶型号混合评测。** base1 身份在场景生成前显式配置，控制使用已准备的 CAD 和标注；不读取本轮仿真型号来选择 CAD，也不读取实时仿真物体位姿来控制。

random 指采用官方 `demo_randomized` 的杂物、背景、光照和桌高设置。保留原任务目标摆放规则；药瓶资产固定 base1。固定指令为 `Pick up the pill bottle and place it onto the blue pad.`，不采用官方 unseen 指令采样，因此不是完整官方随机化 benchmark。

只新增独立场景与批次统计包装，没有修改原药瓶、杯子、微波炉、RoboTwin 或 VLA 源码。沿用原 known-model 入口安装控制器，不将微波炉的接近排序器或取消门板尺寸检查迁移到药瓶。

## 统计和资源

从 seed 0 连续尝试，直到完成 50 个正常初始化的场景。初始化拒绝单独保存并补足；感知、抓取、规划、放置失败全部计入有效场景分母，不替换方法失败。

完整成功要求控制器完成且官方任务评分通过；另报官方评分成功率。时间保留每轮完整阶段计时，成功完成均时按有效场景剔除首轮；如果模型首次加载发生在后续场景，将另行说明冷启动影响，不把包含大额加载的均值直接称为稳态。

GPU 7 仿真/VLA，GPU 6 常驻 FoundationPose/分割视觉；共享 GPU 无算力隔离，JAX platform allocator。模型与感知进程整批复用。结束后的真值位置、接触与官方评分仅用于独立评价，不回传控制。

## 命令

```bash
cd /data1/xuhuixin/ula
GPU=7 PILLBOTTLE_RANDOM_VISION_GPU=6 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
bash scripts/run_move_pillbottle_pad_known_randomized.sh \
  --known-model 1 --known-approach annotated --known-contact-region all \
  --known-approach-search ranked --known-placement-policy visual \
  --episodes 50 --seed 0 \
  --output /data1/xuhuixin/ula/results/pillbottle_known_randomized_50valid_seed0_20261102_v1 \
  > /data1/xuhuixin/ula/results/pillbottle_known_randomized_50valid_seed0_20261102_v1.launcher.log 2>&1
```

复现请更换为空输出目录，不覆盖旧结果。入口的 `--episodes` 表示有效场景目标数，不是只尝试 50 个原始 seed。日志 `[ula-pillbottle-random] valid=.../50` 为已完成有效数量。

## 结果

已启动，尚未结束。最终以 `report.json` 的 `run_status=finished`、`completed_valid_episodes=50` 为准；结果将同时核对源码指纹、工作进程关闭、初始化拒绝和失败分段。
