#!/usr/bin/env python3
"""Build the static ULA meeting report from existing results; never run control.

Requires the existing pi05 Python environment (matplotlib, numpy, Pillow).
Only writes the PDF/DOCX report and docs/figures/ula_group_meeting.
"""

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "figures" / "ula_group_meeting"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "font.size": 15, "axes.unicode_minus": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
BLUE, INK, GRAY = "#086BCA", "#183951", "#60717F"
PALE, EDGE, LINE = "#DEEDF8", "#CFD9E0", "#8B9296"
WARM, WARMBG = "#A45E17", "#FFF3E3"

SOURCES = {
    "A": "results/three_stage_50eps_seed0_profiled/report.json",
    "B": "results/pointcloud_relaxed_12mm_35deg_50eps_seed0_gpu5_v1/report.json",
    "pc90": "results/pointcloud_full_50eps_20261024_190030/report.json",
    "pc92": "results/pointcloud_mv_retry_50eps_seed0_gpu5_v1/report.json",
}
REPORTS = {k: json.loads((ROOT / v).read_text()) for k, v in SOURCES.items()}
STATS = {k: d["timing_statistics"]["successful_excluding_first_episode"]
         for k, d in REPORTS.items() if k in ("A", "B")}
assert REPORTS["A"]["successes"] == 50
assert REPORTS["B"]["successes"] == 48
assert STATS["A"]["samples"] == 49 and STATS["B"]["samples"] == 47
assert abs(STATS["A"]["mean_episode_wall_s"] - 6.905559007) < 1e-6
assert abs(STATS["B"]["mean_episode_wall_s"] - 5.895244081) < 1e-6


def canvas(w=1200, h=700):
    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, size=20, color=INK, ha="left", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, linespacing=1.5, **kw)


def box(ax, x, y, w, h, s, size=21, fill=PALE, color=BLUE, edge=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=22",
                 facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=2))
    label = txt(ax, x+w/2, y+h/2, s, size, color, ha="center", weight="bold", zorder=3)
    # Diagram coordinates are pixels at 100 dpi, while font sizes are points.
    # Fit labels to the actual rendered box, including Chinese glyph metrics.
    bounds = label.get_window_extent(ax.figure.canvas.get_renderer())
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x+w-24, y+h-18))
    scale = min(1., abs(p1[0]-p0[0])/max(bounds.width, 1),
                abs(p1[1]-p0[1])/max(bounds.height, 1))
    label.set_fontsize(size*scale)


def arrow(ax, points, dashed=False, color=LINE):
    if len(points) > 2:
        ax.plot(*zip(*points[:-1]), color=color, lw=1.7,
                ls="--" if dashed else "-", zorder=0)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="->",
                 mutation_scale=17, lw=1.7, color=color,
                 linestyle="--" if dashed else "-", zorder=0))


def note(ax, s, y=658):
    txt(ax, 36, y, s, 14, GRAY)


FIGS = {}


def savefig(name, fig):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=170)
    FIGS[name] = OUT / f"{name}.png"
    plt.close(fig)


# 1: Explain traditional control without assuming robotics knowledge.
f, a = canvas(1200, 430)
for x, title, body in [
    (35, "感知：在哪里？", "RGB-D → 杯子表面与杯垫位置"),
    (440, "几何：手停哪里？", "物体观测 → 末端位置与朝向"),
    (845, "规划：关节怎样转？", "CuRobo → 可执行关节轨迹"),
]:
    box(a, x, 92, 320, 140, title + "\n" + body, 18)
arrow(a, [(355, 162), (440, 162)])
arrow(a, [(760, 162), (845, 162)])
box(a, 210, 293, 780, 80,
    "需要机器人运动学模型 ≠ 必须准备杯子 CAD", 24)
note(a, "传统控制根据当前观测重新计算目标与轨迹，不是固定动作回放。", 405)
savefig("01_control_layers", f)

# 2: Match the user's pale-blue rounded-box, gray-arrow reference.
f, a = canvas(1200, 790)
box(a, 260, 20, 620, 110, "ULA 调度层\n传统接近 → VLA 抓取 → 传统放置", 24)
arrow(a, [(410, 130), (410, 165), (270, 165), (270, 255)])
arrow(a, [(730, 130), (730, 255)])
txt(a, 270, 205, "接近 / 搬运放置", 19, BLUE, ha="center")
txt(a, 730, 205, "局部抓取", 19, BLUE, ha="center")
box(a, 45, 255, 450, 110,
    "RGB-D 感知 + 显式几何\n生成手的目标位置与朝向", 22)
box(a, 525, 255, 410, 110,
    "图像 + 机器人状态 + 语言\nWorldLA 生成关节动作块", 21)
arrow(a, [(270, 365), (270, 430)])
arrow(a, [(730, 365), (730, 430)])
box(a, 45, 430, 450, 110, "CuRobo 运动规划\n末端目标 → 关节运动轨迹", 22)
box(a, 525, 430, 410, 110, "TOPP 时间参数化\n关节动作 → 可执行短轨迹", 21)
arrow(a, [(270, 540), (270, 588), (465, 588), (465, 635)])
arrow(a, [(730, 540), (730, 635)])
box(a, 300, 635, 600, 105,
    "共同执行层\n关节驱动 + 夹爪驱动 + SAPIEN", 23)
arrow(a, [(900, 685), (1105, 685), (1105, 75), (880, 75)])
box(a, 970, 315, 215, 100, "机器人状态\n与新观测反馈", 20, "white", BLUE, BLUE)
note(a, "同一场景连续运行，不重置、不瞬移；抓稳切换仍使用仿真物体位姿与接触信息。", 768)
savefig("02_architecture", f)

# 3: Real evidence, not generated illustrations.
EP = ROOT / Path(SOURCES["B"]).parent / "episode_000_seed0"
f, a = canvas(1200, 750)
phases = [
    ("initial", "① 初始场景", "杯子与杯垫由相机观察"),
    ("handoff", "② 传统接近到位", "末端到达目标，交给 VLA"),
    ("grasp_handoff", "③ VLA 抓稳", "完成闭爪与抬起，交回传统"),
    ("final", "④ 传统放置完成", "搬运、下降、松爪并撤离"),
]
for i, (stem, title, sub) in enumerate(phases):
    x, y = 45 + (i % 2) * 590, 15 + (i // 2) * 355
    img = np.asarray(Image.open(EP / f"{stem}_head_camera.png"))
    a.imshow(img, extent=[x+95, x+415, y+250, y+10], aspect="auto")
    txt(a, x+255, y+283, title, 23, BLUE, ha="center", weight="bold")
    txt(a, x+255, y+321, sub, 17, GRAY, ha="center")
note(a, "真实记录：B 方案 / seed 0 / 头部相机。四幅图来自同一轮任务。", 733)
savefig("03_episode_sequence", f)

# 4: Explicit known-CAD and metadata inputs.
f, a = canvas(1200, 620)
box(a, 35, 40, 310, 100, "已准备杯子 CAD\n与接触 / 功能点标注", 21)
box(a, 35, 210, 310, 100, "当前 RGB-D\n检测 + 分割", 21)
arrow(a, [(345, 90), (420, 90), (420, 180), (455, 180)])
arrow(a, [(345, 260), (420, 260), (420, 210), (455, 210)])
box(a, 455, 145, 305, 125, "FoundationPose\n对齐模型，估计位置与朝向", 20)
arrow(a, [(760, 205), (820, 205)])
box(a, 820, 145, 340, 125, "把预标注参照变到现场\n接近目标 / 放置功能点", 20)
box(a, 95, 405, 1010, 115,
    "传统接近 → VLA 抓取 → 重新感知 → 传统放置\n抓后重新计算杯子与手的保持关系", 22)
arrow(a, [(990, 270), (990, 405)])
note(a, "杯垫目标来自分割与深度，不读取杯垫 CAD；FoundationPose 是“有模型辅助的视觉定位”。", 583)
savefig("04_known_model", f)

# 5: Side-view geometric teaching sketch, explicitly not a reconstruction.
f, a = canvas(1200, 620)
box(a, 35, 35, 300, 100, "RGB-D + 杯子分割\n回投成可见表面点", 20)
arrow(a, [(335, 85), (415, 85)])
box(a, 415, 35, 330, 100, "估计可见范围\n以及局部桌面高度", 20)
arrow(a, [(745, 85), (825, 85)])
box(a, 825, 35, 340, 100, "生成粗接近目标\n不计算精确接触点", 20)
a.plot([90, 690], [485, 485], color=LINE, lw=2)
txt(a, 90, 519, "视觉估计的桌面", 18, GRAY)
t = np.linspace(-1, 1, 25)
for x in [295, 425]:
    a.scatter(np.full_like(t, x), 405 + 70*t, s=13, color=BLUE, alpha=.7)
a.scatter(np.linspace(295, 425, 20), np.full(20, 335), s=13, color=BLUE, alpha=.7)
a.add_patch(Rectangle((295, 335), 130, 150, fill=False, ec=LINE, ls="--", lw=1.4))
a.scatter([295], [240], s=105, color=BLUE, zorder=3)
txt(a, 305, 205, "TCP 接近目标", 21, BLUE)
arrow(a, [(500, 240), (500, 335)])
txt(a, 520, 284, "可见最高点\n上方 10 cm", 19, GRAY)
txt(a, 220, 566, "侧视示意，不代表真实比例或完整杯子形状", 15, GRAY)
box(a, 760, 225, 405, 280,
    "当前左臂的目标规则\nX：靠左臂一侧边界\nY：可见范围中点\nZ：可见最高点 + 10 cm\n朝向：预设向下模板", 21)
note(a, "看不全时可用腕部相机补看；可见范围不是完整物体，也不等于最优抓取姿态。", 600)
savefig("05_pointcloud_approach", f)

# 6: A compact three-stage causal diagram for CAD-free placement.
f, a = canvas(1200, 680)
box(a, 35, 40, 330, 130, "抓前观测\n保存表面点云\n桌面约束 → 支撑锚点", 21)
box(a, 435, 40, 330, 130, "抓后观测\nICP 对齐前后表面\n估计平移与旋转", 21)
box(a, 835, 40, 330, 130, "放置目标\n把锚点对准视觉杯垫\n补偿抓后倾斜", 21)
arrow(a, [(365, 105), (435, 105)])
arrow(a, [(765, 105), (835, 105)])
for cx, cy, tilt in [(195, 395, 0), (590, 355, -.25), (1005, 395, 0)]:
    pts = np.array([[-50, 0], [-50, -115], [50, -115], [50, 0]])
    r = np.array([[np.cos(tilt), -np.sin(tilt)], [np.sin(tilt), np.cos(tilt)]])
    q = pts @ r.T + [cx, cy]
    a.plot(q[:, 0], q[:, 1], color=BLUE, lw=3)
    a.scatter([cx], [cy], s=80, color=BLUE, zorder=3)
a.plot([85, 310], [395, 395], color=LINE, lw=2)
a.plot([885, 1125], [405, 405], color=LINE, lw=2)
a.add_patch(Rectangle((952, 394), 106, 10, color=EDGE))
txt(a, 195, 445, "近似支撑锚点", 21, BLUE, ha="center")
txt(a, 195, 485, "可见 XY 中心 + 桌面高度", 17, GRAY, ha="center")
txt(a, 590, 445, "抓后锚点随估计运动更新", 19, BLUE, ha="center")
txt(a, 1005, 445, "目标：锚点落在杯垫上", 19, BLUE, ha="center")
arrow(a, [(325, 335), (445, 300)])
arrow(a, [(735, 305), (865, 350)])
box(a, 140, 545, 920, 75,
    "关键假设：初始直立 + 水平桌面 + 刚性夹持 + 前后表面有重叠", 21)
note(a, "支撑锚点是几何近似，不是直接看见的杯底，也不是预标注功能点；不回退到 FoundationPose。", 655)
savefig("06_pointcloud_placement", f)

# 7: Keep the truth-dependent gate visible rather than hiding it in notes.
f, a = canvas(1200, 520)
box(a, 35, 70, 330, 110, "传统接近", 26)
box(a, 435, 70, 330, 110, "VLA 局部抓取", 26)
box(a, 835, 70, 330, 110, "传统放置", 26)
arrow(a, [(365, 125), (435, 125)])
arrow(a, [(765, 125), (835, 125)])
box(a, 60, 250, 485, 160,
    "第一次交接：检查末端到位\n规划执行成功\n位置 ≤2 cm，朝向 ≤15°", 22)
box(a, 640, 250, 500, 160,
    "第二次交接：检查抓稳\n抬升 ≥3 cm + 接触 + 夹爪状态\n连续 3 步相对平移稳定 ≤1 cm", 21,
    WARMBG, WARM, "#E5CFB1")
arrow(a, [(400, 125), (400, 215), (300, 215), (300, 250)])
arrow(a, [(800, 125), (800, 215), (890, 215), (890, 250)])
note(a, "蓝框：机器人状态与视觉目标。浅橙框：目前读取仿真杯子位姿和物理接触，影响控制切换。", 475)
savefig("07_handoff", f)

# 8: Plot the original disjoint stage means, not values copied from prose.
f, a = canvas(1200, 610)
colors = ["#96C6EA", "#176CC0", "#619ED3", "#D9E1E6"]
labels = ["传统接近", "VLA 抓取", "传统放置", "公共开销"]
for idx, key in enumerate(("A", "B")):
    st = STATS[key]
    vals = [st["stages"][k]["wall_s"] for k in
            ("traditional_approach", "vla_rollout", "traditional_placement")]
    vals.append(st["mean_episode_wall_s"] - sum(vals))
    y = 160 + idx*190
    txt(a, 40, y-43, "A 已知模型" if key == "A" else "B 无预备杯子模型", 21, BLUE, weight="bold")
    x = 40
    for v, c in zip(vals, colors):
        w = v*137
        a.add_patch(Rectangle((x, y), w, 65, color=c))
        txt(a, x+w/2, y+32, f"{v:.3f}", 18,
            "white" if c in colors[1:3] else INK, ha="center", weight="bold")
        x += w
    txt(a, x+20, y+30, f"{st['mean_episode_wall_s']:.3f} s", 23, BLUE, weight="bold")
    txt(a, 40, y+92,
        "50/50 成功（100%） · 时间 n=49" if key == "A" else "48/50 成功（96%） · 时间 n=47", 18, GRAY)
for i, (lab, c) in enumerate(zip(labels, colors)):
    x = 60 + i*280
    a.add_patch(Rectangle((x, 505), 22, 22, color=c))
    txt(a, x+35, 516, lab, 18)
note(a, "时间：排除首轮，仅成功轮平均；样本集合不同，不是单模块严格因果消融。", 575)
savefig("08_results", f)

# 9: Separate iteration regressions and gate failures.
f, a = canvas(1200, 610)
for x, rate, title, sub in [
    (30, "90%", "点云完整流程", "45/50\n原配准门槛"),
    (435, "92%", "+ 补观测 / 条件恢复", "46/50\n恢复实际触发 0 次"),
    (840, "96%", "+ 放宽配准拒绝门槛", "48/50\n12 mm / 35°"),
]:
    box(a, x, 60, 330, 220, "", 20)
    txt(a, x+165, 110, rate, 40, BLUE, ha="center", weight="bold")
    txt(a, x+165, 170, title, 20, BLUE, ha="center", weight="bold")
    txt(a, x+165, 235, sub, 18, GRAY, ha="center")
arrow(a, [(360, 170), (435, 170)])
arrow(a, [(765, 170), (840, 170)])
box(a, 40, 350, 530, 135, "剩余失败 seed 11\n抓稳与配准通过\n21 组搬运 / 放置候选均不可行", 21)
box(a, 630, 350, 530, 135, "剩余失败 seed 42\n传统接近未达到 2 cm 容差\n未启动 VLA，不强制交接", 21)
note(a, "92% → 96%：33/48 从配准拒绝变成功；无新增失败。seed 48 成功后仍倾斜约 5.56°。", 550)
note(a, "同一组 seeds 的迭代回归结果，不是独立泛化测试；不能把收益归因于未触发的重抓。", 587)
savefig("09_iterations", f)

# 10: An information-source boundary, useful for research claims.
f, a = canvas(1200, 550)
box(a, 40, 50, 530, 130,
    "视觉目标生成：不读物体真值\n杯子范围 / 桌面 / 杯垫\n在线点云配准与放置几何", 22)
box(a, 630, 50, 530, 130,
    "机器人侧：仍需已知信息\n运动学模型 / 相机标定\n关节与末端状态", 22)
box(a, 40, 250, 530, 130,
    "控制交接：仍读物体真值\n杯子抬升 / 相对平移稳定\n物理接触检测", 22,
    WARMBG, WARM, "#E5CFB1")
box(a, 630, 250, 530, 130,
    "最终评测：使用物体真值\n原任务成功 / 位置误差\n事后最终倾斜评测", 22,
    "#F3F6F8", GRAY, EDGE)
note(a, "无预备杯子 CAD ≠ 无机器人模型 ≠ 全流程无真值 ≠ 已证明未知杯子泛化。", 450)
note(a, "仿真资产用于渲染与物理；控制器不读取其几何先验。这两个层面需要分开。", 500)
savefig("10_truth_boundary", f)


def table(ax, rows, widths, x=55, y=155, row_h=48, size=17):
    """Controlled layout: predictable cell extents and legible Chinese text."""
    for r, row in enumerate(rows):
        xx = x
        for c, cell in enumerate(row):
            ax.add_patch(Rectangle((xx, y+r*row_h), widths[c], row_h,
                         facecolor=PALE if r == 0 else ("#F5F8FA" if r % 2 else "white"),
                         edgecolor="white", lw=2))
            txt(ax, xx+14, y+(r+.5)*row_h, str(cell), size,
                BLUE if r == 0 else INK, weight="bold" if r == 0 else "normal")
            xx += widths[c]


def page(title, sub, num):
    f, a = canvas(1440, 810)
    txt(a, 55, 53, title, 30, BLUE, weight="bold")
    txt(a, 57, 100, sub, 17, GRAY)
    a.plot([55, 1385], [125, 125], color=EDGE, lw=1.5)
    txt(a, 55, 778, "ULA 组会汇报  |  2026.10.24  |  基于现有实验结果", 13, GRAY)
    txt(a, 1385, 778, f"{num:02d} / 14", 13, GRAY, ha="right")
    return f, a


def embed(f, name, bounds):
    ax = f.add_axes(bounds)
    ax.imshow(Image.open(FIGS[name]))
    ax.axis("off")


def takeaway(ax, s, y=714):
    box(ax, 55, y-27, 1330, 55, s, 19)


def bullet(ax, y, title, body, x=65, size=21):
    txt(ax, x, y, title, size, BLUE, weight="bold")
    txt(ax, x, y+44, body, size-2, INK)


pages = []
f, a = page("ULA：让传统控制与 VLA 各做擅长的部分", "从已知杯子模型辅助，走向无预备杯子几何的三阶段协作", 1)
box(a, 70, 175, 1300, 105, "传统控制接近 → VLA 局部抓取 → 传统控制搬运与放置", 29)
table(a, [["当前完整任务结果", "已知模型辅助 A", "无预备杯子模型 B"],
          ["成功率（含全部 50 轮）", "100% · 50/50", "96% · 48/50"],
          ["稳态成功轮平均完成时间", "6.906 s · n=49", "5.895 s · n=47"]],
      [520, 400, 410], y=345, row_h=65, size=22)
txt(a, 75, 610, "研究问题：同一个场景里，怎样可靠地完成两次控制权交接？", 25, BLUE, weight="bold")
txt(a, 75, 662, "时间排除首轮；B 仍用仿真真值判断抓稳。当前不是未见杯子泛化实验。", 20, GRAY)
takeaway(a, "本次先证明“接力流程可运行”，还不能证明融合普遍优于纯传统或纯 VLA。")
pages.append(f)

f, a = page("01  传统控制在做什么？", "用三个问题理解：在哪里？手停哪里？关节怎样转？", 2)
embed(f, "01_control_layers", [.025, .37, .95, .47])
bullet(a, 555, "末端 EE / 工具中心 TCP", "分别对应手腕控制坐标系与夹爪工作位置，需要用机器人运动学进行转换。")
txt(a, 65, 653, "VLA 的不同：从图像、状态和语言直接预测关节动作，而不是先手工写出每个目标。", 21)
takeaway(a, "传统控制不等于依赖杯子 CAD；机器人模型与物体几何先验是两件事。")
pages.append(f)

f, a = page("02  ULA 总体架构", "沿用同一机器人、同一物体状态、同一仿真场景；切换控制者，不切换任务现场", 3)
embed(f, "02_architecture", [.11, .10, .78, .73])
pages.append(f)

f, a = page("03  一次真实任务的四个时刻", "无预备杯子模型方案 B · seed 0 · 原始相机记录", 4)
embed(f, "03_episode_sequence", [.16, .09, .68, .75])
pages.append(f)

f, a = page("04  方案 A：已知模型辅助的视觉定位", "FoundationPose 把已知杯子模型对齐到当前 RGB-D，不是自动生成未知杯子的模型", 5)
embed(f, "04_known_model", [.045, .21, .91, .60])
takeaway(a, "接近参照来自预标注接触点；放置参照来自预标注功能点。抓取动作仍由 VLA 完成。")
pages.append(f)

f, a = page("05  方案 B 接近：量出眼前可见范围", "不读取杯子 CAD、尺寸或接触点；先给 VLA 一个粗接近起点", 6)
embed(f, "05_pointcloud_approach", [.045, .21, .91, .60])
takeaway(a, "这不是精确抓取点求解：传统控制送到附近，最后接触与闭爪交给 VLA。")
pages.append(f)

f, a = page("06  方案 B 放置：用前后点云追踪刚体运动", "ICP 可以理解为：平移、旋转两堆测量点，让共同可见的表面尽量贴合", 7)
embed(f, "06_pointcloud_placement", [.06, .20, .88, .62])
takeaway(a, "支撑锚点是近似，不是真实杯底；对称、遮挡或夹持滑移都可能让推断失效。")
pages.append(f)

f, a = page("07  交接怎么判断？加载开销怎么处理？", "到位后才让 VLA 启动；抓稳后立即截断动作块，交回传统控制", 8)
embed(f, "07_handoff", [.045, .30, .91, .49])
txt(a, 65, 603, "模型常驻：VLA 与视觉各加载一次，每次重新看图；每轮只清空观测和动作缓存。", 22)
txt(a, 65, 651, "B 本批：109 次新帧请求；首轮 61.221 s 含加载 / 预热，稳态统计剔除首轮。", 21, GRAY)
takeaway(a, "抓稳交接目前仍依赖仿真物体位姿与接触，这是走向实机前要替换的关键环节。")
pages.append(f)

f, a = page("08  实验设置与统计口径", "50 次不是 50 种杯子，而是同一杯子在 50 个种子下的初始摆放", 9)
table(a, [
    ["项目", "当前设置"],
    ["平台 / 任务", "RoboTwin · place_empty_cup · demo_clean"],
    ["场景 / 机器人", "aloha-agilex；固定左臂及对应侧采样；默认光栅渲染"],
    ["VLA", "原 WorldLA 100000 步权重；未为融合重新训练"],
    ["推理 / 预算", "tail_blend=0.25；推理 seed 20260901；总预算 500 动作"],
    ["B 额外设置", "启用腕部补观测与至多一次条件恢复；ICP 12 mm / 35°"],
    ["成功标准", "功能点 XY <3.5 cm、Z <1.5 cm，双夹爪打开；不含直立角度"],
    ["统计", "成功率含全部 50 轮；时间排除首轮，只平均成功轮次"],
    ["计时 / 硬件", "主机墙钟，含环境与截图；A100-SXM4-80GB / GPU 5"],
], [300, 1030], row_h=55, size=19)
takeaway(a, "固定 clean 配置与仿真交接条件，不等同于官方全任务评测或实机效果。")
pages.append(f)

f, a = page("09  主结果：VLA 基本不变，省时主要在传统两段", "两组均使用相同 WorldLA；A 的完成均值 n=49，B 为 n=47", 10)
embed(f, "08_results", [.045, .20, .91, .62])
takeaway(a, "两组均值相差约 1.010 s（14.6%），但成功样本不同，不能当成纯删模块的因果增益。")
pages.append(f)

f, a = page("10  B 的时间分解：大头不只在神经网络", "排除首轮后的 47 个成功样本；下列为互不重叠的分项，单位秒", 11)
stage = STATS["B"]["stages"]
ap = stage["traditional_approach"]["components_s"]
pl = stage["traditional_placement"]["components_s"]
v = stage["vla_rollout"]["components_s"]
def rem(d, keys):
    return sum(d.values()) - sum(d.get(k, 0) for k in keys)
tradkeys = ["observation", "perception", "pointcloud_registration", "motion_planning", "trajectory_execution"]
left = [["传统分项", "接近", "放置"]]
for label, key in zip(["观测获取", "视觉端到端", "点云配准", "运动规划", "轨迹执行"], tradkeys):
    left.append([label, f"{ap.get(key, 0):.3f}", f"{pl.get(key, 0):.3f}"])
left += [["目标生成 / 其余", f"{rem(ap, tradkeys):.3f}", f"{rem(pl, tradkeys):.3f}"],
         ["合计", f"{sum(ap.values()):.3f}", f"{sum(pl.values()):.3f}"]]
table(a, left, [310, 130, 130], y=168, row_h=56, size=19)
vkeys = ["observation", "input_preprocessing", "model_inference", "trajectory_time_parameterization", "action_execution"]
right = [["VLA 分项", "耗时"]]
for label, key in zip(["观测获取", "输入预处理", "模型采样 / 回传", "TOPP 时间参数化", "动作执行 / 仿真"], vkeys):
    right.append([label, f"{v[key]:.3f}"])
right += [["后处理 / 其余", f"{rem(v, vkeys):.3f}"], ["合计", f"{sum(v.values()):.3f}"]]
table(a, right, [420, 180], x=760, y=168, row_h=56, size=19)
txt(a, 65, 663, "TOPP + 动作执行占 VLA 阶段约 73.6%；单纯优化模型推理不能消除全部耗时。", 22, BLUE)
takeaway(a, "这些是当前仿真 / 控制接口的墙钟时间，不是实机完成动作所需时间；分项不可重复加总。")
pages.append(f)

f, a = page("11  迭代结果与剩余失败", "配准拒绝：看起来不够可信而不放行；规划失败：没有找到可执行的关节轨迹", 12)
embed(f, "09_iterations", [.045, .22, .91, .59])
takeaway(a, "放宽质量门槛并未改变 ICP 优化算法；当前不是继续放宽就能解决所有失败。")
pages.append(f)

f, a = page("12  研究边界：无杯子 CAD 不等于全系统无真值", "必须区分目标生成、控制切换与事后评测使用了什么信息", 13)
embed(f, "10_truth_boundary", [.045, .28, .91, .52])
txt(a, 65, 648, "仍未证明：未见杯子泛化、精确直立放置、未知障碍安全避障、实机部署能力。", 22, WARM)
takeaway(a, "当前结论：无预备杯子几何的目标生成已跑通；纯视觉自主交接尚未完成。")
pages.append(f)

f, a = page("13  结论与下一步", "先把“可靠接力”讲清楚，再检验融合是否带来可重复的性能优势", 14)
bullet(a, 185, "已经完成", "同场景双交接；常驻模型；分段计时；无预备杯子模型完成 48/50 次完整任务。", size=23)
bullet(a, 295, "近期：针对两类剩余失败", "核查范围估计与接近目标；改进抓后可达性判断和放置候选，而非只放宽 ICP。", size=23)
bullet(a, 405, "中期：减少真值依赖、提高放置质量", "视觉 / 力觉抓稳判断；末段视觉修正；新增独立直立角度与放置稳定性指标。", size=23)
bullet(a, 515, "验证：冻结参数后的新对象与严格基线", "新 seeds、新杯子、多次重复；补充同条件纯传统 / 纯 VLA / 融合对照与消融。", size=23)
txt(a, 65, 640, "数据出处：three_stage_50eps_seed0_profiled / pointcloud_relaxed_12mm_35deg_50eps_seed0_gpu5_v1", 15, GRAY)
takeaway(a, "组会讨论：下一阶段更聚焦“弱先验的可靠交接”，还是“融合策略的性能优势”？")
pages.append(f)

pdf_path = DOCS / "ula_group_meeting_report.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.infodict().update({"Title": "ULA：传统控制与 VLA 分阶段协作组会汇报",
                          "Author": "ULA", "Subject": "现有实验结果与架构说明"})
    for i, f in enumerate(pages, 1):
        pdf.savefig(f)
        f.savefig(OUT / f"page_{i:02d}.png", dpi=90)
        plt.close(f)

# Contact sheet for a visual check of every page, not just the title page.
thumbs = []
for i in range(1, len(pages)+1):
    im = Image.open(OUT / f"page_{i:02d}.png").convert("RGB")
    im.thumbnail((720, 405))
    thumbs.append(im)
sheet = Image.new("RGB", (1440, 405*((len(thumbs)+1)//2)), "#D3DCE2")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i%2)*720, (i//2)*405))
sheet.save(OUT / "page_contact_sheet.png")

audit = {
    "date": "2026-10-24", "pdf_pages": len(pages),
    "statistics": {k: {"successes": REPORTS[k]["successes"],
                       "total_episodes": len(REPORTS[k]["episodes"]),
                       "completion_time_samples": STATS[k]["samples"],
                       "mean_completion_s": STATS[k]["mean_episode_wall_s"]}
                   for k in ("A", "B")},
    "sources": {k: {"path": str(ROOT / v),
                    "sha256": hashlib.sha256((ROOT/v).read_bytes()).hexdigest()}
                for k, v in SOURCES.items()},
    "note": "Existing result files read only; no experiment launched. SVG text remains editable.",
}
(OUT / "report_sources.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
print(f"Built {pdf_path} ({len(pages)} pages)")
print(f"Figures and previews: {OUT}")


def build_docx():
    """Portable editable narrative version, using standard Office Open XML.

    Text and tables remain editable; the diagram PNGs are embedded, not linked.
    This avoids adding dependencies to the robot/VLA environments.
    """
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    }
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)
    def node(parent, name, attrs=None, text=None):
        prefix, local = name.split(":")
        attrs = {f"{{{ns[k.split(':')[0]]}}}{k.split(':')[1]}" if ":" in k else k: str(v)
                 for k, v in (attrs or {}).items()}
        el = ET.SubElement(parent, f"{{{ns[prefix]}}}{local}", attrs)
        if text is not None:
            el.text = text
        return el
    def xml(el):
        return ET.tostring(el, encoding="utf-8", xml_declaration=True)
    pkgrel = "http://schemas.openxmlformats.org/package/2006/relationships"
    rels = ET.Element("Relationships", xmlns=pkgrel)
    def rel(kind, target, external=False):
        rid = f"rId{len(rels)+1}"
        attrs = {"Id": rid, "Type": ns["r"]+"/"+kind, "Target": target}
        if external:
            attrs["TargetMode"] = "External"
        ET.SubElement(rels, "Relationship", attrs)
        return rid
    rel("styles", "styles.xml")
    doc = ET.Element(f"{{{ns['w']}}}document")
    body = node(doc, "w:body")
    styles = ET.Element(f"{{{ns['w']}}}styles")
    defaults = node(node(styles, "w:docDefaults"), "w:rPrDefault")
    rp = node(defaults, "w:rPr")
    node(rp, "w:rFonts", {"w:ascii": "Calibri", "w:hAnsi": "Calibri", "w:eastAsia": "微软雅黑"})
    node(rp, "w:sz", {"w:val": 22})
    for style_id, name, size, color in [
        ("Normal", "Normal", 22, "183951"),
        ("Title", "Title", 38, "086BCA"),
        ("Heading1", "heading 1", 30, "086BCA"),
        ("Heading2", "heading 2", 25, "086BCA"),
        ("Code", "Code", 17, "334B5C"),
        ("Caption", "Caption", 18, "60717F"),
    ]:
        s = node(styles, "w:style", {"w:type": "paragraph", "w:styleId": style_id})
        node(s, "w:name", {"w:val": name})
        pp = node(s, "w:pPr")
        spacing = {"w:after": 120, "w:line": 320, "w:lineRule": "auto"}
        if style_id.startswith("Heading") or style_id == "Title":
            node(pp, "w:keepNext")
            spacing["w:before"] = 220
            if style_id.startswith("Heading"):
                node(pp, "w:outlineLvl", {"w:val": 0 if style_id == "Heading1" else 1})
        node(pp, "w:spacing", spacing)
        rp = node(s, "w:rPr")
        node(rp, "w:sz", {"w:val": size})
        node(rp, "w:color", {"w:val": color})
        if style_id in ("Title", "Heading1", "Heading2"):
            node(rp, "w:b")
        if style_id == "Code":
            node(rp, "w:rFonts", {"w:ascii": "Consolas", "w:hAnsi": "Consolas", "w:eastAsia": "微软雅黑"})
    def run(p, text, bold=False, code=False):
        rr = node(p, "w:r")
        if bold or code:
            props = node(rr, "w:rPr")
            if bold:
                node(props, "w:b")
            if code:
                node(props, "w:rFonts", {"w:ascii": "Consolas", "w:hAnsi": "Consolas"})
        t = node(rr, "w:t", text=text)
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    def para(parent, text="", style="Normal"):
        p = node(parent, "w:p")
        node(node(p, "w:pPr"), "w:pStyle", {"w:val": style})
        parts = re.split(r"(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                run(p, part[2:-2], bold=True)
            elif part.startswith("`") and part.endswith("`"):
                run(p, part[1:-1], code=True)
            elif re.fullmatch(r"\[[^\]]+\]\([^)]+\)", part):
                label, target = re.match(r"\[([^\]]+)\]\(([^)]+)\)", part).groups()
                # Resolve local references so links do not assume Word's CWD.
                if not re.match(r"https?://", target):
                    target = (DOCS / target).resolve().as_uri()
                hl = node(p, "w:hyperlink", {"r:id": rel("hyperlink", target, True)})
                run(hl, label)
            else:
                run(p, part)
        return p
    images = {}
    def picture(path, caption):
        number = len(images)+1
        image_name = f"figure_{number:02d}.png"
        images[image_name] = path
        rid = rel("image", "media/"+image_name)
        im = Image.open(path)
        cx, cy = 5852160, int(5852160*im.height/im.width)
        p = node(body, "w:p")
        node(node(p, "w:pPr"), "w:jc", {"w:val": "center"})
        inline = node(node(node(p, "w:r"), "w:drawing"), "wp:inline",
                      {"distT": 0, "distB": 0, "distL": 0, "distR": 0})
        node(inline, "wp:extent", {"cx": cx, "cy": cy})
        node(inline, "wp:docPr", {"id": number, "name": image_name, "descr": caption})
        graphic = node(inline, "a:graphic")
        gd = node(graphic, "a:graphicData", {"uri": ns["pic"]})
        pic = node(gd, "pic:pic")
        nv = node(pic, "pic:nvPicPr")
        node(nv, "pic:cNvPr", {"id": 0, "name": image_name})
        node(nv, "pic:cNvPicPr")
        fill = node(pic, "pic:blipFill")
        node(fill, "a:blip", {"r:embed": rid})
        node(node(fill, "a:stretch"), "a:fillRect")
        sp = node(pic, "pic:spPr")
        transform = node(sp, "a:xfrm")
        node(transform, "a:off", {"x": 0, "y": 0})
        node(transform, "a:ext", {"cx": cx, "cy": cy})
        node(node(sp, "a:prstGeom", {"prst": "rect"}), "a:avLst")
        p = para(body, f"图 {number}  {caption}", "Caption")
        node(p.find(f"{{{ns['w']}}}pPr"), "w:jc", {"w:val": "center"})
    def word_table(rows):
        tb = node(body, "w:tbl")
        prop = node(tb, "w:tblPr")
        node(prop, "w:tblW", {"w:w": 9216, "w:type": "dxa"})
        node(prop, "w:tblLayout", {"w:type": "fixed"})
        borders = node(prop, "w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            node(borders, "w:"+edge, {"w:val": "single", "w:sz": 4, "w:color": "D5E1EA"})
        grid = node(tb, "w:tblGrid")
        n = len(rows[0])
        weights = [1.5]+[1]*(n-1)
        widths = [int(9216*w/sum(weights)) for w in weights]
        for width in widths:
            node(grid, "w:gridCol", {"w:w": width})
        for i, row in enumerate(rows):
            tr = node(tb, "w:tr")
            trp = node(tr, "w:trPr")
            node(trp, "w:cantSplit")
            if i == 0:
                node(trp, "w:tblHeader")
            for j, text in enumerate(row):
                tc = node(tr, "w:tc")
                tcp = node(tc, "w:tcPr")
                node(tcp, "w:tcW", {"w:w": widths[j], "w:type": "dxa"})
                if i == 0:
                    node(tcp, "w:shd", {"w:fill": "DEEDF8"})
                    text = "**"+text+"**"
                para(tc, text)
        para(body)
    lines = (DOCS / "ula_group_meeting_report.md").read_text().splitlines()
    code, i = False, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            code = not code
        elif code:
            para(body, line, "Code")
        elif not line.strip():
            pass
        elif line.startswith("!["):
            caption, path = re.match(r"!\[([^\]]+)\]\(([^)]+)\)", line).groups()
            picture(DOCS/path, caption)
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [cell.strip() for cell in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in row):
                    rows.append(row)
                i += 1
            word_table(rows)
            continue
        elif line.startswith("# "):
            para(body, line[2:], "Title")
        elif line.startswith("## "):
            para(body, line[3:], "Heading1")
        else:
            para(body, "• "+line[2:] if line.startswith("- ") else line)
        i += 1
    sect = node(body, "w:sectPr")
    node(sect, "w:pgSz", {"w:w": 11906, "w:h": 16838})
    node(sect, "w:pgMar", {"w:top": 1050, "w:right": 1345, "w:bottom": 1050, "w:left": 1345,
                           "w:header": 500, "w:footer": 500, "w:gutter": 0})
    content = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    for ext, mime in [("rels", "application/vnd.openxmlformats-package.relationships+xml"),
                      ("xml", "application/xml"), ("png", "image/png")]:
        ET.SubElement(content, "Default", Extension=ext, ContentType=mime)
    for part, suffix in [("document", "document.main"), ("styles", "styles")]:
        ET.SubElement(content, "Override", PartName=f"/word/{part}.xml",
                      ContentType=f"application/vnd.openxmlformats-officedocument.wordprocessingml.{suffix}+xml")
    rootrels = ET.Element("Relationships", xmlns=pkgrel)
    ET.SubElement(rootrels, "Relationship", Id="rId1", Type=ns["r"]+"/officeDocument", Target="word/document.xml")
    target = DOCS / "ula_group_meeting_report.docx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, el in {"[Content_Types].xml": content, "_rels/.rels": rootrels,
                         "word/document.xml": doc, "word/styles.xml": styles,
                         "word/_rels/document.xml.rels": rels}.items():
            z.writestr(name, xml(el))
        for name, path in images.items():
            z.write(path, "word/media/"+name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(z.read(name))
    assert len(images) == 10
    print(f"Built editable narrative {target} ({len(images)} embedded figures)")


build_docx()
