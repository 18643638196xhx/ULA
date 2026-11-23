# 新服务器恢复步骤

## 0. 目录约定

```
<ULA>     = 你的 ULA 根目录（示例 /data1/<user>/ula）
<ROBOTWIN>= <ULA>/tra_pipe/RoboTwin
```

## 1. 拉 ULA 代码

```bash
git clone https://github.com/18643638196xhx/ULA.git <ULA>
cd <ULA>
```

仓库里已经包含（无需额外操作）：

| 内容 | 路径 |
|---|---|
| 全部执行器 / runner / 脚本 | `adapters/` `scripts/` `rules/` |
| 36 个编译好的规则 | `rule_snapshots/<task>/<arm>.yaml` |
| 规则的输入（视觉轨迹 / 模板 / 感知） | `derived/` |
| 794 局评测摘要 + 成功率总表 | `eval_reports/` `EVAL_SUMMARY.md` |
| RoboTwin 的任务代码与配置（含我们的补丁） | `environment/robotwin/` |
| 依赖清单 | `environment/requirements_pi05.txt` |
| 标定文件 | `configs/robot_frames/aloha_agilex_frames_20261121_v2.json` |

## 2. 拉 RoboTwin（官方）并打补丁

```bash
cd <ULA>/tra_pipe
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f     # 必须钉住，否则行为不一致

# 下载仿真资产（16 GB，按官方文档）
bash assets/_download.py

# 覆盖成我们的版本（关键，包含 force_arm / 渲染 / 规划器补丁）
cp -r <ULA>/environment/robotwin/envs/*        envs/
cp -r <ULA>/environment/robotwin/env_cfg/*     env_cfg/
cp -r <ULA>/environment/robotwin/description/* description/
```

补丁要点（`environment/robotwin/envs/` 内已含）：

- `envs/_base_task.py`、`envs/camera/camera.py`、`envs/robot/planner.py`、`envs/robot/robot.py`
- `envs/place_empty_cup.py`（`force_arm` 支持）

## 3. Python 环境

```bash
cd <ULA>
uv venv external/pi05/.venv --python 3.11
source external/pi05/.venv/bin/activate
pip install -r environment/requirements_pi05.txt
```

清单里包含 git 依赖（会自动拉取）：

```
curobo   @ git+https://github.com/NVlabs/curobo.git@ebb71702...
openpi   @ git+https://github.com/RoboTwin-Platform/RoboTwin.git@50ea3a80.../policy/pi05
```

## 4. 拉 WorldLA 代码与权重

### 4.1 代码

```bash
cd <ULA>/external
git clone <你的 WorldLA 代码仓库> wam_vla
git -C wam_vla checkout <钉住的 commit>
```

> ULA 通过 `external/wam_vla` 导入 `experiments.oracle_shared_kv_multi_task4.common`。

### 4.2 权重（HuggingFace）

```bash
pip install -U huggingface_hub
hf download xhx0327/WorldLA-100k-decay-full --local-dir /data1/<user>/worldla_hf
```

目录对应关系：

| 下载到 | 指向 | 说明 |
|---|---|---|
| `worldla_hf/merged/params` | `checkpoints/worldla_100k/action_checkpoints/final_step_100000` | Orbax 格式动作模型（181 MB）|
| `worldla_hf/merged/assets` | `checkpoints/worldla_assets/assets` | 含 `paligemma_tokenizer.model` + `robotwin_official50_demo_clean_v21` |

```bash
cd <ULA>
mkdir -p checkpoints
ln -s <路径>/merged/params  checkpoints/worldla_100k_link   # 见下方说明
```

> **注意**：代码默认 `checkpoints/worldla_100k` 是「包含 `action_checkpoints/final_step_100000` 的目录」。
> 新服务器上建议直接建软链，把 `merged/params` 放到该层级：

```bash
mkdir -p checkpoints/worldla_100k/action_checkpoints
ln -s $(realpath worldla_hf/merged/params) checkpoints/worldla_100k/action_checkpoints/final_step_100000
ln -s $(realpath worldla_hf/merged/assets) checkpoints/worldla_assets/assets
```

### 4.3 tokenizer 缓存

```bash
mkdir -p <ULA>/.runtime_cache/openpi/big_vision
cp worldla_hf/merged/assets/paligemma_tokenizer.model <ULA>/.runtime_cache/openpi/big_vision/
```

## 5. 冒烟验证

```bash
cd <ULA>
GPU=0 bash scripts/run_compiled_rule.sh \
  --rule rule_snapshots/place_empty_cup/left.yaml \
  --calibration configs/robot_frames/aloha_agilex_frames_20261121_v2.json \
  --output results/smoke_cup \
  --task-config demo_randomized --episodes 2 --seed 0
```

预期：`task_success` 至少 1/2（参考基线 98%）。

## 6. 缺失项（无法从 GitHub 恢复）

| 项 | 影响 | 补救 |
|---|---|---|
| 194 条原始示范（90 GB）| 不能重训 VLA；不能换算法重提规则 | 需要重新采集（`collect_configs/` 里有配置）|
| 16 GB 仿真资产 | 无法运行 | 官方 `assets/_download.py` |
| `third_party/`（2.2 GB）| 部分任务可能缺依赖 | 官方仓库自带 |

已有 `derived/`（tracks + 模板 + 感知）时，**规则可以重建、执行可以复现**，只是不能训练。
