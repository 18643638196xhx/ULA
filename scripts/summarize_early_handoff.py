#!/usr/bin/env python3
"""Read completed early-handoff and historical baseline results, write a comparison."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def compare(experiment, baseline):
    new = json.loads((experiment/"report.json").read_text())
    old = json.loads(baseline.read_text())
    a = {e["seed"]: e for e in old["episodes"]}
    b = {e["seed"]: e for e in new["episodes"]}
    first_a, first_b = old["episodes"][0]["seed"], new["episodes"][0]["seed"]
    common = sorted(s for s in a.keys() & b.keys() if a[s]["success"] and b[s]["success"]
                    and s not in (first_a, first_b))
    gains = sorted(s for s in a.keys() & b.keys() if b[s]["success"] and not a[s]["success"])
    regressions = sorted(s for s in a.keys() & b.keys() if a[s]["success"] and not b[s]["success"])
    def metrics(report):
        episodes = report["episodes"]
        success = [e for e in episodes[1:] if e["success"]]
        stages = ("traditional_approach", "vla_rollout", "traditional_grasp_verification", "traditional_placement")
        values = {k: mean(e["timing_profile"]["stages"].get(k, {}).get("wall_s", 0) for e in success)
                  for k in stages} if success else {}
        duration = mean(e["elapsed_s"] for e in success) if success else None
        if success:
            values["other"] = duration-sum(values.values())
        return dict(total=len(episodes), successes=sum(bool(e["success"]) for e in episodes),
            success_rate=sum(bool(e["success"]) for e in episodes)/len(episodes),
            completion_samples=len(success), mean_completion_s=duration,
            mean_attempt_s=mean(e["elapsed_s"] for e in episodes[1:]) if len(episodes)>1 else None,
            mean_vla_steps_success=mean(e["vla_steps"] for e in success) if success else None,
            stages_success=values)
    contact = [e for e in new["episodes"] if e.get("contact_handoff")]
    # Avoid interpreting an unverified contact as a successful grasp.
    verified = [e for e in contact if e.get("grasp_handoff") and e["grasp_handoff"]["ready"]]
    def failure_stage(e):
        return e.get("failure_stage") or "task_evaluation"
    result = dict(baseline=str(baseline.resolve()), experiment=str(experiment.resolve()),
                  old=metrics(old), new=metrics(new), common_success_seeds=common,
                  gains=gains, regressions=regressions,
                  contact_handoffs=len(contact), verified_trial_lifts=len(verified),
                  contacts_before_3cm=sum(e["contact_handoff"]["cup_lift_m"]<.03 for e in contact),
                  failure_stages=dict(Counter(failure_stage(e) for e in new["episodes"] if not e["success"])))
    if common:
        result["paired_common_success"] = dict(samples=len(common),
            old_mean_s=mean(a[s]["elapsed_s"] for s in common),
            new_mean_s=mean(b[s]["elapsed_s"] for s in common),
            old_vla_steps=mean(a[s]["vla_steps"] for s in common),
            new_vla_steps=mean(b[s]["vla_steps"] for s in common))
    (experiment/"comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    def fmt(x):
        return "无样本" if x is None else f"{x:.6f}"
    lines = ["# 提前交接完整任务对照", "", "## 主结果", "",
             "| 指标 | 历史稳定抬升交接 | 本次闭爪交接 + 传统试抬 |", "| --- | ---: | ---: |"]
    for label, key in [("完整任务成功", "successes"), ("总轮数", "total"),
                       ("排除首轮后的成功时间样本数", "completion_samples")]:
        lines.append(f"| {label} | {result['old'][key]} | {result['new'][key]} |")
    for label, key in [("成功率", "success_rate"), ("成功轮平均完成时间，s", "mean_completion_s"),
                       ("全部尝试平均时间，s", "mean_attempt_s"), ("成功轮平均 VLA 动作数", "mean_vla_steps_success")]:
        old_value, new_value = result['old'][key], result['new'][key]
        if key == "success_rate":
            lines.append(f"| {label} | {old_value:.1%} | {new_value:.1%} |")
        else:
            lines.append(f"| {label} | {fmt(old_value)} | {fmt(new_value)} |")
    lines += ["", "成功率包含首轮及所有失败；其余均值排除首轮。失败不从分母剔除。", "",
              "## 成功轮时间分解，排除首轮", "", "| 阶段，s | 历史 | 本次 |", "| --- | ---: | ---: |"]
    for key, name in [("traditional_approach", "传统接近"), ("vla_rollout", "VLA"),
                      ("traditional_grasp_verification", "传统试抬与验证"),
                      ("traditional_placement", "传统放置"), ("other", "环境、截图与其余开销")]:
        lines.append(f"| {name} | {fmt(result['old']['stages_success'].get(key))} | {fmt(result['new']['stages_success'].get(key))} |")
    lines += ["", "## 交接与失败", "",
              f"- 接触交接 {len(contact)}/{len(b)}，其中 {result['contacts_before_3cm']} 轮交接时抬升未满 3 cm。",
              f"- 传统试抬最终验证通过 {len(verified)} 轮；这不代表后续放置全部成功。",
              f"- 相对历史：新成功 seeds {gains}；新失败 seeds {regressions}。",
              f"- 失败阶段：{result['failure_stages']}。", ""]
    for e in new["episodes"]:
        if not e["success"]:
            lines.append(f"- seed {e['seed']}：{failure_stage(e)}；{e.get('error') or '控制序列执行结束，但原任务成功评测未通过'}")
            ev = e.get("evaluation_only", {})
            if "final_xy_error_m" in ev:
                lines.append(f"  最终水平误差 {ev['final_xy_error_m']*1000:.3f} mm，"
                             f"高度差 {ev['final_z_error_m']*1000:.3f} mm。")
    if common:
        p = result["paired_common_success"]
        lines += ["", "## 共同成功样本对照", "",
                  f"排除两组首轮后，共同成功 {p['samples']} 个 seeds：{common}。",
                  f"历史平均 {p['old_mean_s']:.6f} s，本次 {p['new_mean_s']:.6f} s；",
                  f"平均 VLA 动作数 {p['old_vla_steps']:.3f} → {p['new_vla_steps']:.3f}。"]
    lines += ["", "## 解释边界", "",
              "这是和已完成历史批次的同 seed 对照，不是同期随机交错实验；不能排除运行波动。",
              "接触与试抬验证仍读取仿真物体位姿/接触。夹爪值是现有控制状态，不是力觉传感器。",
              "传统试抬有新开销，且本次额外接触时刻截图计入总时间，不从统计中扣除。",
              "原任务成功不含精确直立角度；单个 seed 的抓后状态可能随重跑变化。",
              "独立先导不计入本批结果；原始结果与配置见 report.json 和 EXPERIMENT.md。"]
    (experiment/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT/"results/pointcloud_relaxed_12mm_35deg_50eps_seed0_gpu5_v1/report.json")
    args = parser.parse_args()
    compare(args.experiment, args.baseline)
