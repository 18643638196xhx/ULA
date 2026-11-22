#!/usr/bin/env python3
"""Build the current two-scheme meeting report, without modifying old reports.

Reuses the existing report's drawing and editable OOXML helpers by loading only
their function definitions (never its top-level experiment/report generation).
All experiment files are read-only; no robot/model evaluation is launched.
"""

import ast
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STEM = "ula_group_meeting_current"
OUT = DOCS / "figures" / STEM
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "font.size": 15, "axes.unicode_minus": False, "svg.fonttype": "none",
    "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})
BLUE, INK, GRAY = "#086BCA", "#183951", "#60717F"
PALE, EDGE, LINE = "#DEEDF8", "#CFD9E0", "#8B9296"
FIGS = {}


class CurrentDocument(ast.NodeTransformer):
    """Retarget ONLY the reused DOCX function to this report and six figures."""
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in (
                "ula_group_meeting_report.md", "ula_group_meeting_report.docx"):
            node.value = node.value.replace("ula_group_meeting_report", STEM)
        return node

    def visit_Assert(self, node):
        if ast.unparse(node.test) == "len(images) == 10":
            node.test = ast.parse("len(images) == 6", mode="eval").body
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
                and isinstance(node.test.func.value, ast.Name)
                and node.test.func.value.id == "line" and node.test.func.attr == "startswith"
                and len(node.test.args) == 1 and isinstance(node.test.args[0], ast.Constant)
                and node.test.args[0].value == "## "):
            node.test.args[0].value = "##"
            node.body = ast.parse('para(body, line.lstrip("# "), "Heading2" if line.startswith("### ") else "Heading1")').body
        return node


source = ROOT / "scripts/build_group_meeting_report.py"
selected = []
for item in ast.parse(source.read_text()).body:
    if isinstance(item, ast.FunctionDef) and item.name in {
            "canvas", "txt", "box", "arrow", "note", "table", "build_docx"}:
        selected.append(CurrentDocument().visit(item) if item.name == "build_docx" else item)
assert len(selected) == 7
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"))

SOURCES = {"A": ROOT / "results/three_stage_50eps_seed0_profiled/report.json",
           "B": ROOT / "results/visual_trial_50eps_20261024_v1/report.json"}
reports = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
stats = {key: report["timing_statistics"]["successful_excluding_first_episode"]
         for key, report in reports.items()}
assert reports["A"]["successes"] == 50 and reports["B"]["successes"] == 49
assert stats["A"]["samples"] == 49 and stats["B"]["samples"] == 48
assert abs(stats["A"]["mean_episode_wall_s"]-6.905559007192449) < 1e-9
assert abs(stats["B"]["mean_episode_wall_s"]-6.119689460960217) < 1e-9
baseline = {"traditional": {"time_s": 5.240, "success_rate": .84},
            "vla": {"time_s": 17.592, "success_rate": .98}}


def save(name, fig):
    for suffix in ("png", "svg"):
        fig.savefig(OUT/f"{name}.{suffix}", dpi=150)
    FIGS[name] = OUT/f"{name}.png"
    plt.close(fig)


# Pale-blue rounded boxes and gray arrows follow the user's reference style.
f, a = canvas(1200, 750)
box(a, 235, 18, 620, 100, "ULA 调度层\n传统接近 → VLA 局部操作 → 传统放置", 23)
arrow(a, [(370, 118), (370, 165), (265, 165), (265, 230)])
arrow(a, [(725, 118), (725, 230)])
txt(a, 265, 190, "接近 / 试抬 / 搬运放置", 18, BLUE, ha="center")
txt(a, 725, 190, "最后接近 / 调整 / 闭爪", 18, BLUE, ha="center")
box(a, 35, 230, 460, 105, "RGB-D + 显式几何\n算出手的目标位置和朝向", 22)
box(a, 525, 230, 400, 105, "RGB + 机器人状态 + 语言\nWorldLA 预测关节动作块", 21)
arrow(a, [(265, 335), (265, 395)])
arrow(a, [(725, 335), (725, 395)])
box(a, 35, 395, 460, 95, "CuRobo\n目标位姿 → 关节轨迹", 22)
box(a, 525, 395, 400, 95, "TOPP\n关节动作 → 可执行短轨迹", 21)
arrow(a, [(265, 490), (265, 545), (430, 545), (430, 595)])
arrow(a, [(725, 490), (725, 595)])
box(a, 280, 595, 640, 95, "共同执行层\n关节驱动 + 夹爪驱动 + SAPIEN", 23)
arrow(a, [(920, 643), (1090, 643), (1090, 68), (855, 68)])
box(a, 955, 275, 225, 95, "机器人状态\n与新观测反馈", 19, "white", BLUE, BLUE)
note(a, "A：真值辅助抓稳交接。B：闭爪候选 + 视觉试抬交接；物体真值只用于最终评分。", 724)
save("01_architecture", f)

f, a = canvas(1200, 590)
box(a, 35, 40, 300, 100, "预备杯子 CAD\n抓取接触点 / 放置功能点", 20)
box(a, 35, 210, 300, 100, "当前 RGB-D\nGroundingDINO + SAM2", 20)
arrow(a, [(335, 90), (395, 90), (395, 177), (435, 177)])
arrow(a, [(335, 260), (395, 260), (395, 210), (435, 210)])
box(a, 435, 145, 310, 125, "FoundationPose\n把已知形状对齐到当前画面", 20)
arrow(a, [(745, 205), (815, 205)])
box(a, 815, 145, 350, 125, "当前杯子位置与朝向\n把模型标注变成现场目标", 20)
box(a, 75, 365, 1050, 100, "传统接近 → VLA 闭爪并抬起 → 传统搬运放置\n抓后重新视觉定位，更新杯子与手的保持关系", 22)
arrow(a, [(990, 270), (990, 365)])
note(a, "A 的抓稳交接读取杯子位姿 / 物理接触真值；杯垫目标来自视觉深度，不需杯垫 CAD。", 545)
save("02_known_model", f)

f, a = canvas(1200, 610)
box(a, 35, 30, 330, 110, "RGB-D + 检测分割\n可见杯身表面 → 本轮点云", 21)
box(a, 435, 30, 330, 110, "粗接近目标\n可见侧边 / 中点 / 顶部 +10 cm", 20)
box(a, 835, 30, 330, 110, "CuRobo 接近\n最后细调、闭爪交给 VLA", 20)
arrow(a, [(365, 85), (435, 85)])
arrow(a, [(765, 85), (835, 85)])
box(a, 35, 245, 330, 130, "初始视觉点云\n可见 XY 中心 + 桌面高度\n定义近似支撑锚点", 21)
box(a, 435, 245, 330, 130, "抓后重新看\nICP 配准共同可见表面\n估计杯子平移与旋转", 21)
box(a, 835, 245, 330, 130, "传统放置\n锚点对准视觉杯垫\n补偿估计倾斜后松爪", 21)
arrow(a, [(365, 310), (435, 310)])
arrow(a, [(765, 310), (835, 310)])
box(a, 100, 455, 1000, 85, "不读杯子 CAD / 标注，不调用 FoundationPose\n仍假设初始直立、水平桌面、刚性夹持与可见表面重叠", 21)
note(a, "点云是实际看见的表面，不是完整杯子；近似锚点不是重建出来的隐藏杯底。", 585)
save("03_pointcloud", f)

f, a = canvas(1200, 650)
for x, title in [(35, "VLA 闭爪候选"), (435, "传统试抬"), (835, "视觉跟手确认")]:
    box(a, x, 45, 330, 95, title, 24)
arrow(a, [(365, 92), (435, 92)])
arrow(a, [(765, 92), (835, 92)])
box(a, 35, 210, 330, 160, "闭爪控制状态 <0.05\n连续 3 个动作\n只允许尝试，不证明抓稳", 21)
box(a, 435, 210, 330, 160, "先保存可信 RGB-D 基准\n再向上 5 cm，分 3 段\n机器人状态检查末端到位", 21)
box(a, 835, 210, 330, 160, "匹配前后表面，估计上移\n至少 3 cm，方向与大小一致\n通过后才进入传统放置", 20)
for x in (200, 600, 1000):
    arrow(a, [(x, 140), (x, 210)])
box(a, 90, 455, 1020, 95, "观测不可信：有限次重新分割 / 匹配腕部视角\n仍不通过：停止，不进入放置、不主动释放", 22)
note(a, "不读取杯子位姿与物理接触，不恢复严格滑移门槛；视觉启发式不等于物理抓稳证明。", 605)
save("04_visual_handoff", f)

# Raw values are directly labeled. Baseline provenance is visible on the chart.
f, ax = plt.subplots(1, 2, figsize=(12, 4.9))
labels = ["纯传统*", "纯 VLA*", "A 已知模型", "B 当前方案"]
times = [5.240, 17.592, stats["A"]["mean_episode_wall_s"], stats["B"]["mean_episode_wall_s"]]
rates = [84, 98, 100, 98]
colors = ["#B8C7D2", "#B8C7D2", "#7DAFDC", BLUE]
for axis, values, title, unit, limit in zip(ax, [times, rates],
        ["平均完成时间", "完整任务成功率"], ["秒", "%"], [21, 115]):
    bars = axis.bar(np.arange(4), values, color=colors, width=.6)
    for index, (bar, value) in enumerate(zip(bars, values)):
        axis.text(bar.get_x()+bar.get_width()/2, value+limit*.025,
            f"{value:.3f}" if unit == "秒" else f"{value}%", ha="center", fontsize=14, color=INK)
        if index < 2:
            bar.set_hatch("//")
    axis.set_xticks(np.arange(4), labels, fontsize=12)
    axis.set_ylim(0, limit)
    axis.set_title(title, fontsize=19, color=BLUE, pad=15)
    axis.set_ylabel(unit, fontsize=13)
    axis.spines[["top", "right"]].set_visible(False)
f.text(.05, .035, "* 用户提供参考基线，样本量与计时口径待核对。A/B 时间排除首轮，仅成功样本：n=49 / 48。", fontsize=11, color=GRAY)
f.subplots_adjust(left=.07, right=.98, bottom=.18, top=.84, wspace=.3)
save("05_results", f)

f, a = canvas(1200, 720)
ep = SOURCES["B"].parent/"episode_000_seed0"
for i, (stem, title) in enumerate([("initial", "初始"), ("handoff", "传统接近到位"),
                                 ("grasp_handoff", "传统试抬、视觉通过"), ("final", "最终成功")]):
    x = 20+i*300
    a.imshow(Image.open(ep/f"{stem}_head_camera.png"), extent=[x, x+280, 270, 60])
    txt(a, x+140, 307, title, 18, BLUE, ha="center")
txt(a, 30, 25, "成功样本 seed 0：同一轮任务的原始头部相机画面", 18, GRAY)
bad = reports["B"]["episodes"][30]
assert bad["seed"] == 30 and not bad["success"]
a.imshow(Image.open(SOURCES["B"].parent/"episode_030_seed30/final_head_camera.png"),
         extent=[100, 460, 650, 380])
box(a, 560, 395, 590, 220, "唯一失败 seed 30\n视觉交接通过，放置动作执行完\n最终水平偏差 4.165 cm >3.5 cm\n高度偏差 0.830 cm <1.5 cm", 22)
note(a, "动作执行完 ≠ 任务成功。失败的直接条件已确认；具体误差来源仍需定位。", 695)
save("06_evidence", f)


def page(title, subtitle, number):
    f, a = canvas(1440, 810)
    txt(a, 55, 50, title, 29, BLUE, weight="bold")
    txt(a, 57, 98, subtitle, 17, GRAY)
    a.plot([55, 1385], [125, 125], color=EDGE, lw=1.5)
    txt(a, 55, 780, "ULA 组会汇报 | 两种融合方案 | 2026.10.24", 13, GRAY)
    txt(a, 1385, 780, f"{number:02d} / 10", 13, GRAY, ha="right")
    return f, a


def embed(f, name, bounds=(.05, .15, .9, .67)):
    axis = f.add_axes(bounds)
    axis.imshow(Image.open(FIGS[name]))
    axis.axis("off")


def takeaway(a, text):
    box(a, 55, 702, 1330, 52, text, 19)


pages = []
f, a = page("ULA：传统控制与 VLA 的分阶段协作", "从已知杯子模型辅助，走向视觉点云与自主交接；本次只比较两种融合实现", 1)
box(a, 70, 170, 1300, 95, "传统接近 → VLA 局部操作 → 传统搬运与放置", 29)
table(a, [["参考 / 实测方法", "平均完成时间", "成功率"],
    ["纯传统 pipeline（用户提供）", "5.240 s", "84%"],
    ["纯 VLA（用户提供）", "17.592 s", "98%"],
    ["A：已知杯子模型 + VLA", "6.906 s · n=49", "100% · 50/50"],
    ["B：当前视觉点云 + VLA", "6.120 s · n=48", "98% · 49/50"]],
    [670, 330, 330], y=320, row_h=59, size=21)
txt(a, 65, 657, "A/B 均时排除首轮，仅成功样本；纯基线样本量和计时口径未核对，不作严格消融。", 19, GRAY)
takeaway(a, "研究问题：如何让两个控制者在同一任务中可靠接力，而不是重新训练一个包办全程的模型？")
pages.append(f)

f, a = page("01 总体架构：调度分工，底层执行共用", "传统控制算目标并规划；VLA 根据图像预测动作。始终是同一机器人、同一场景。", 2)
embed(f, "01_architecture", (.08, .07, .84, .76))
pages.append(f)

f, a = page("02 方案 A：先对齐已知杯子模型", "FoundationPose 做有模型的视觉定位，接近和放置目标利用已有标注", 3)
embed(f, "02_known_model", (.05, .22, .9, .58))
takeaway(a, "A：50/50 成功；抓稳交接仍读取杯子位姿与接触真值，不是纯视觉自主交接。")
pages.append(f)

f, a = page("03 方案 B：测量眼前表面，不准备杯子 CAD", "可见范围用于粗接近；初始点云 + 抓后点云用于估计运动与近似支撑点", 4)
embed(f, "03_pointcloud", (.05, .20, .9, .61))
takeaway(a, "不读杯子 CAD、尺寸与标注，不回退 FoundationPose；可见点云不等于完整形状重建。")
pages.append(f)

f, a = page("04 当前交接：闭爪只是候选，视觉跟手才放行", "VLA 不负责试抬；传统试抬后确认杯子跟手上移，再进入搬运放置", 5)
embed(f, "04_visual_handoff", (.05, .19, .9, .62))
takeaway(a, "控制阶段不读物体位姿或接触真值，并禁用环境真值提前成功终止；只在结束后评分。")
pages.append(f)

f, a = page("05 两种融合方案到底改了什么？", "不是只换掉一个感知模型：几何先验、VLA 截断时机与交接证据都改变了", 6)
table(a, [["环节", "A 已知模型", "B 当前方案"],
    ["杯子先验", "CAD + 接触点 / 功能点标注", "本轮可见 RGB-D 点云"],
    ["接近目标", "标注经视觉位姿变换", "可见范围 + 固定接近规则"],
    ["VLA 负责", "最后接近、闭爪、抬起", "最后接近、闭爪"],
    ["抓后交接", "位姿 / 接触真值 + 稳定性", "闭爪候选 + 视觉试抬"],
    ["放置参照", "当前视觉位姿 + 模型功能点", "点云运动 + 近似支撑锚点"],
    ["真值边界", "交接 / 在线成功终止 / 评分", "仅最终评分"],
    ["共同依赖", "机器人模型 / 状态、相机标定", "同左；保留桌面碰撞先验"]],
    [220, 555, 555], y=155, row_h=62, size=19)
takeaway(a, "无预备杯子模型 ≠ 无机器人模型；减少在线物体真值依赖 ≠ 已完成真机部署。")
pages.append(f)

f, a = page("06 实验设置与统计口径", "50 次是同一杯子资产在不同 seed 下的摆放，不是 50 种未知杯子", 7)
table(a, [["项目", "两种融合方案的共同设置 / 差异"],
    ["任务与场景", "RoboTwin · place_empty_cup · demo_clean；固定左臂；默认光栅渲染"],
    ["样本与模型", "seeds 0–49；同一 WorldLA 100000 步权重，未为融合重新训练"],
    ["推理与预算", "tail_blend=0.25；每块最多 10 动作；每轮总预算 500 动作"],
    ["B 的观测 / 配准", "有界重观测与匹配腕部参考；ICP 12 mm / 35°；本批未触发重抓"],
    ["成功标准", "杯子 / 杯垫功能点 XY <3.5 cm、Z <1.5 cm；双夹爪打开"],
    ["成功率 / 时间", "全 50 次计成功率；排除首轮后只平均成功样本，A n=49 / B n=48"],
    ["时间范围", "主机墙钟：环境建立至最终评分 / 截图；不是实机动作时长"],
    ["纯基线来源", "用户提供 5.240 s / 84%、17.592 s / 98%；样本量与计时口径待核对"]],
    [290, 1040], y=151, row_h=58, size=18)
takeaway(a, "原成功标准不检查精确直立角度；对照与泛化结论必须限定在已核对的设置内。")
pages.append(f)

f, a = page("07 结果：效率与成功率的折中潜力", "纯传统 / 纯 VLA 为参考结果；两种融合由已有原始报告读取，不编造基线样本量", 8)
embed(f, "05_results", (.05, .28, .9, .51))
txt(a, 65, 633, "A：100% / 6.906 s；B：98% / 6.120 s。均时样本不同，不能把差值全归因于去掉 CAD。", 21)
takeaway(a, "结果支持可行性；尚不能据单批次和未统一口径的参考数据宣称统计显著优势。")
pages.append(f)

f, a = page("08 分段时间：B 把试抬从 VLA 移给传统控制", "排除首轮后的成功样本；互不重叠的阶段耗时，单位秒；A n=49，B n=48", 9)
stage_keys = ["traditional_approach", "vla_rollout", "traditional_grasp_verification", "traditional_placement"]
rows = [["阶段", "A 已知模型", "B 当前方案"]]
for label, key in zip(["传统接近", "VLA 局部阶段", "传统试抬 + 视觉确认", "传统搬运放置"], stage_keys):
    rows.append([label]+[f"{stats[k]['stages'][key]['wall_s']:.3f}" if key in stats[k]["stages"]
                        else "不单设，VLA 已完成抬起" for k in ("A", "B")])
remaining = {k: stats[k]["mean_episode_wall_s"]-sum(stats[k]["stages"].get(s,{}).get("wall_s",0)
                for s in stage_keys) for k in ("A", "B")}
rows.extend([["环境、截图、评分及其余"]+[f"{remaining[k]:.3f}" for k in ("A", "B")],
             ["总计（未舍入均值）"]+[f"{stats[k]['mean_episode_wall_s']:.3f}" for k in ("A", "B")]])
table(a, rows, [570, 380, 380], y=163, row_h=63, size=22)
txt(a, 65, 655, "B 含显式视觉试抬成本；VLA 提前闭爪截断，阶段缩短。舍入后相加可能相差 0.001 s。", 20, GRAY)
takeaway(a, "下一步需统一四种方法的评测设置，再定位最终落点误差，冻结参数后验证新杯子。")
pages.append(f)

f, a = page("09 任务证据、剩余失败与结论", "B：50 次视觉交接均通过，完整任务成功 49 次；唯一失败是 seed 30 最终水平偏差超限", 10)
embed(f, "06_evidence", (.13, .18, .74, .63))
takeaway(a, "已验证固定任务下的无在线物体真值交接；未知杯子、真实传感噪声与真机可靠性仍待验证。")
pages.append(f)

assert len(pages) == 10 and len(FIGS) == 6
with PdfPages(DOCS/f"{STEM}.pdf") as pdf:
    pdf.infodict().update(Title="ULA：已知杯子模型与当前视觉交接方案", Author="ULA")
    for i, f in enumerate(pages, 1):
        pdf.savefig(f)
        f.savefig(OUT/f"page_{i:02d}.png", dpi=85)
        plt.close(f)

sheet = Image.new("RGB", (1440, 405*5), "#D3DCE2")
for i in range(10):
    image = Image.open(OUT/f"page_{i+1:02d}.png").convert("RGB")
    image.thumbnail((720, 405))
    sheet.paste(image, ((i%2)*720, (i//2)*405))
sheet.save(OUT/"page_contact_sheet.png")
build_docx()

audit = dict(date="2026-10-24", pdf_pages=10, embedded_figures=6,
    user_provided_baselines=baseline,
    baseline_caveat="No original logs or sample counts specified; timing comparability not verified",
    reports={k: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        successes=reports[k]["successes"], total=len(reports[k]["episodes"]),
        completion_samples=stats[k]["samples"], mean_completion_s=stats[k]["mean_episode_wall_s"])
        for k, path in SOURCES.items()},
    note="Only A and current B; no intermediate variants; no evaluation launched; old reports retained")
(OUT/"report_sources.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
print(f"Built {DOCS/STEM}.md / .docx / .pdf; previews: {OUT}")
