# 运行环境快照

复现 ULA 所需的环境信息。只收代码与版本，不收大文件。

## 已收录

| 路径 | 内容 | 大小 |
|---|---|---|
| `robotwin/envs/` | RoboTwin 全部任务实现（51 个任务 + utils/robot/camera） | 600 KB |
| `robotwin/env_cfg/` | 环境与采集配置（含 `task_config/*.yml`） | 228 KB |
| `robotwin/description/` | 任务指令文本（`task_instruction/`） | 3.1 MB |
| `requirements_pi05.txt` | `pip freeze`（301 个包） | — |
| `versions.md` | Python / Torch / JAX / CUDA / GPU 驱动版本 | — |

## 未收录（体积原因）

| 项 | 大小 | 说明 |
|---|---|---|
| `robotwin/assets/` | 16 GB | 仿真资产（网格 + 材质），随 RoboTwin 官方发布 |
| `robotwin/envs/curobo/` | 239 MB | 第三方运动规划库（CuRobo），按官方说明安装 |
| `external/pi05/.venv/` | 数 GB | Python 虚拟环境，用 `requirements_pi05.txt` 重建 |
| WorldLA 权重 | — | 模型 checkpoint，见 `adapters/worldla.py` 的默认路径 |

## 关键版本

```
python  3.11.14
torch   2.6.0+cu124  (CUDA 12.4)
jax     0.5.0
GPU     NVIDIA A100-SXM4-80GB, driver 550.127.08
```

## 重建步骤

```bash
# 1) RoboTwin 官方代码与资产
git clone <robotwin-official> tra_pipe/RoboTwin
# 2) 用本仓库的代码覆盖任务实现与配置
cp -r environment/robotwin/envs/*    tra_pipe/RoboTwin/envs/
cp -r environment/robotwin/env_cfg/* tra_pipe/RoboTwin/env_cfg/
cp -r environment/robotwin/description/* tra_pipe/RoboTwin/description/
# 3) CuRobo 按官方文档安装到 envs/curobo
# 4) Python 环境
uv venv external/pi05/.venv --python 3.11
pip install -r environment/requirements_pi05.txt
# 5) 运行
GPU=0 bash scripts/run_compiled_rule.sh --rule rule_snapshots/<task>/left.yaml ...
```
