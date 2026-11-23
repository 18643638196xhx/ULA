# 新服务器恢复步骤

> 目标：在新机器上把 ULA 跑起来（规则可执行、WorldLA 可推理）。
> 分三部分：代码、依赖、权重。**第 5 节的路径改写是最容易踩的坑。**

---

## 1. 拉 ULA 代码

```bash
git clone https://github.com/18643638196xhx/ULA.git ula
cd ula
```

仓库里已包含：

| 内容 | 路径 |
|---|---|
| 执行器 / runner / 脚本 | `adapters/` `scripts/` `rules/` |
| **54 个规则版本**（含 handover_block 全部 9 版） | `rule_snapshots/`（见 `MANIFEST.json`） |
| 规则的输入（视觉轨迹 37 / 多视角模板 11 / 感知 59） | `derived/tracks` `derived/templates` `derived/perception` |
| **50 任务点云模板**（感知必需） | `derived/pointcloud_templates/<task>/template.npz` |
| RoboTwin 任务代码与配置（含我们的补丁） | `environment/robotwin/` |
| 依赖清单 | `environment/requirements_pi05.txt` |
| WorldLA 运行时配置 | `environment/worldla/experiment.json` 等 |
| 标定文件 | `configs/robot_frames/aloha_agilex_frames_20261121_v2.json` |
| 评测摘要与成功率表 | `eval_reports/` `EVAL_SUMMARY.md` |
| 示范数据清单（sha256/大小/臂别） | `demo_manifest.json` |

---

## 2. 拉 RoboTwin（官方）并打补丁

```bash
cd ula/tra_pipe
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f   # 必须钉住

bash assets/_download.py                                # 16 GB 资产

cp -r ../../environment/robotwin/envs/*        envs/
cp -r ../../environment/robotwin/env_cfg/*     env_cfg/
cp -r ../../environment/robotwin/description/* description/
```

补丁内容（已在 `environment/robotwin/envs/` 内）：

- `_base_task.py`、`camera/camera.py`、`robot/planner.py`、`robot/robot.py`
- `place_empty_cup.py`（`force_arm` 支持）

---

## 3. Python 环境

```bash
cd ula
uv venv external/pi05/.venv --python 3.11
source external/pi05/.venv/bin/activate
pip install -r environment/requirements_pi05.txt
```

该清单含 git 依赖，会自动拉：

```
curobo  @ git+https://github.com/NVlabs/curobo.git@ebb71702...
openpi  @ git+https://github.com/RoboTwin-Platform/RoboTwin.git@50ea3a80.../policy/pi05
```

---

## 4. 拉 WorldLA 代码与权重

### 4.1 代码

```bash
cd ula/external
git clone https://github.com/18643638196xhx/WorldLA.git wam_vla
```

> ULA 通过 `external/wam_vla` 导入 `experiments.oracle_shared_kv_multi_task4.common`。

### 4.2 权重

```bash
pip install -U huggingface_hub
hf download xhx0327/WorldLA-100k-decay-full --local-dir /data1/<user>/worldla_hf
```

下载后结构：

```
worldla_hf/merged/    ← 推荐使用（LoRA 已合并，自包含）
    params/           Orbax 参数
    assets/robotwin_official50_demo_clean_v21/   + paligemma_tokenizer.model
    config.json  example_inference.py  README.md
worldla_hf/original/  ← 仅用于复现参考
```

---

## 5. 搭出 ULA 期望的路径（关键）

ULA 代码里写死两个路径：

```python
DEFAULT_WORLDLA_MODEL_DIR  = <ula>/checkpoints/worldla_100k
DEFAULT_WORLDLA_ASSET_CKPT = <ula>/checkpoints/worldla_assets
```

而 `wam_vla` 期望的 `model_dir` 是**训练输出格式**：

```
model_dir/
  experiment.json              ← 里面有绝对路径，必须改写
  action_checkpoints/final_step_100000/   ← Orbax 参数（即 HF 的 merged/params）
```

### 5.1 建目录

```bash
cd ula
mkdir -p checkpoints/worldla_100k/action_checkpoints
cp environment/worldla/experiment.json checkpoints/worldla_100k/
ln -s $(realpath /data1/<user>/worldla_hf/merged/params) \
      checkpoints/worldla_100k/action_checkpoints/final_step_100000

mkdir -p checkpoints/worldla_assets
ln -s $(realpath /data1/<user>/worldla_hf/merged/assets) checkpoints/worldla_assets/assets
```

### 5.2 改写 experiment.json 里的绝对路径

旧机器上的路径是 `/data2/xuhuixin/wam_vla_outputs/...`，新机器不存在：

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("checkpoints/worldla_100k/experiment.json")
d = json.loads(p.read_text())
root = pathlib.Path("checkpoints/worldla_100k").resolve()
d["adapter_params"] = str(root / "action_checkpoints/final_step_100000")
d["asset_dir"]      = str(pathlib.Path("checkpoints/worldla_assets/assets").resolve())
p.write_text(json.dumps(d, indent=2))
print("adapter_params =", d["adapter_params"])
print("asset_dir      =", d["asset_dir"])
PY
```

### 5.3 tokenizer 缓存

```bash
mkdir -p ula/.runtime_cache/openpi/big_vision
cp /data1/<user>/worldla_hf/merged/assets/paligemma_tokenizer.model \
   ula/.runtime_cache/openpi/big_vision/
```

---

## 6. 冒烟验证

```bash
cd ula
GPU=0 bash scripts/run_compiled_rule.sh \
  --rule rule_snapshots/place_empty_cup/place_empty_cup_rgbd_left_v1.yaml \
  --calibration configs/robot_frames/aloha_agilex_frames_20261121_v2.json \
  --output results/smoke_cup \
  --task-config demo_randomized --episodes 2 --seed 0
```

验收标准：

| 检查 | 期望 |
|---|---|
| WorldLA 能加载 | 日志出现 `[stage-router] condition=wam_stage_dynamic` |
| 感知能跑 | `perception/estimate.npz` 有点云 |
| 任务成功率 | ≥ 1/2（基线 98%） |

---

## 7. 无法从仓库恢复的东西

| 项 | 影响 | 补救 |
|---|---|---|
| 194 条原始示范（90 GB）| 不能重训 VLA；不能换算法重提规则 | 用 `collect_configs/` 重采 |
| 16 GB 仿真资产 | 完全跑不了 | `assets/_download.py` |
| `third_party/`（2.2 GB）| 部分任务缺依赖 | RoboTwin 官方仓库自带 |

**有 `derived/` 就能重建规则、复现执行**，只是不能训练。
