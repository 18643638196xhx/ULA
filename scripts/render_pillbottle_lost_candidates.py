#!/usr/bin/env python3
"""Post-hoc evidence pictures from saved RGB-D + frozen candidate summaries.

No model inference or controller changes. Unselected masks were not archived:
draw the ORIGINAL search boxes and projected measured centres, never invent
segmentation masks or treat a nearby centre as independent identity proof.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GREEN = (0, 210, 110)
RED = (255, 70, 70)
YELLOW = (255, 232, 0)
DARK = (22, 32, 48)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(None)
def font(size):
    return ImageFont.truetype('/usr/share/fonts/SourceHanSansSC-Regular.otf', size)


def project(point, intrinsic, extrinsic):
    transform = np.asarray(extrinsic, float)
    if transform.shape == (3, 4):
        transform = np.vstack((transform, [0, 0, 0, 1]))
    point = transform[:3, :3] @ np.asarray(point, float) + transform[:3, 3]
    if not np.isfinite(point).all() or point[2] <= 0:
        return None
    uv = np.asarray(intrinsic, float) @ point
    return (uv[:2] / uv[2]).tolist()


def text(draw, xy, value, size=22, fill=DARK):
    draw.text(xy, value, font=font(size), fill=fill)


def badge(draw, xy, value, color):
    x, y = xy
    bounds = draw.textbbox((x, y), value, font=font(20))
    draw.rectangle((x-3, y-2, bounds[2]+4, bounds[3]+3), fill=DARK)
    text(draw, (x, y), value, 20, color)


def annotated(rgb, truth, candidates):
    # Yellow is explicitly an evaluation-only label, not a model prediction.
    outline = truth.copy()
    interior = np.zeros_like(truth)
    interior[1:-1, 1:-1] = (truth[1:-1, 1:-1] & truth[:-2, 1:-1]
        & truth[2:, 1:-1] & truth[1:-1, :-2] & truth[1:-1, 2:])
    outline &= ~interior
    raw = rgb.copy()
    raw[outline] = YELLOW
    image = Image.fromarray(raw).resize((640, 480), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    sx, sy = 640/rgb.shape[1], 480/rgb.shape[0]
    for candidate in reversed(candidates):
        color = GREEN if candidate['position_close_only'] else RED
        x0, y0, x1, y1 = np.asarray(candidate['search_box_xyxy']) * [sx, sy, sx, sy]
        x0, x1 = np.clip([x0, x1], 0, 639)
        y0, y1 = np.clip([y0, y1], 0, 479)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"C{candidate['rank']}"
        badge(draw, (min(x0+5, 580), min(y0+4, 445)), label, color)
        if candidate['projected_center_xy'] is not None:
            x, y = np.asarray(candidate['projected_center_xy']) * [sx, sy]
            if 0 <= x < 640 and 0 <= y < 480:
                draw.ellipse((x-6, y-6, x+6, y+6), fill=DARK, outline=color, width=3)
                draw.line((x-10, y, x+10, y), fill=color, width=2)
                draw.line((x, y-10, x, y+10), fill=color, width=2)
    return image


def render_case(rgb, truth, row, target, intrinsic, extrinsic):
    reference = np.asarray(row['unified']['object']['reference_position_world'])
    candidates = []
    for rank, group in enumerate(target['candidates'], 1):
        if len(group['candidates']) != 1:
            raise ValueError('This gallery expects one saved initial head view')
        value = group['candidates'][0]
        error = float(np.linalg.norm(np.asarray(value['position_world']) - reference))
        candidates.append(dict(rank=rank, search_box_xyxy=value['box_xyxy'],
            position_world=value['position_world'],
            projected_center_xy=project(value['position_world'], intrinsic, extrinsic),
            reference_center_error_m=error, position_close_only=error <= .02,
            ranking_score=group['mean_evidence_ranking_score'],
            semantic_score=group['mean_semantic_score'],
            identity_independently_verified=False))
    if not any(c['position_close_only'] for c in candidates):
        raise ValueError('The previous claim of a nearby candidate is not supported')
    selected_rank = None
    if target['required_available'] and target['evidence']:
        actual = np.asarray(target['evidence'][0]['position_world'])
        selected_rank = min(candidates, key=lambda c: np.linalg.norm(np.asarray(c['position_world'])-actual))['rank']
    status = '候选冲突，拒绝输出' if not target['required_available'] else f'误选了 C{selected_rank}'
    overlay = annotated(rgb, truth, candidates)
    tile = Image.new('RGB', (640, 640), '#f2f5f8')
    draw = ImageDraw.Draw(tile)
    text(draw, (12, 7), f"seed {row['seed']}｜{status}", 23)
    tile.paste(overlay, (0, 48))
    for index, candidate in enumerate(candidates):
        color = (0, 115, 55) if candidate['position_close_only'] else (185, 25, 25)
        line = (f"C{candidate['rank']}：排序分 {candidate['ranking_score']:.3f}；"
                f"中心差 {candidate['reference_center_error_m']*1000:.1f} mm")
        text(draw, (12, 533+index*25), line, 20, color)
    card = Image.new('RGB', (1280, 780), 'white')
    draw = ImageDraw.Draw(card)
    text(draw, (18, 8), f"药瓶 seed {row['seed']}｜旧版通过｜统一版：{status}", 29)
    text(draw, (18, 58), '原始 RGB（无标注）', 23)
    text(draw, (658, 58), '保存的候选框、中心投影与事后标签', 23)
    card.paste(Image.fromarray(rgb).resize((640, 480), Image.Resampling.NEAREST), (0, 96))
    card.paste(overlay, (640, 96))
    for index, candidate in enumerate(candidates):
        color = (0, 115, 55) if candidate['position_close_only'] else (185, 25, 25)
        line = (f"C{candidate['rank']}：系统排序分 {candidate['ranking_score']:.3f}；"
                f"三维可见中心与标签参考相差 {candidate['reference_center_error_m']*1000:.1f} mm")
        text(draw, (18, 587+index*29), line, 22, color)
    text(draw, (18, 714), '绿：位置接近的候选；红：其他候选；黄：独立标签轮廓。排序分不是概率。', 22)
    text(draw, (18, 746), '框是原始搜索区域，不是分割边界；位置接近不等于候选身份或完整分割已通过核验。', 21)
    return card, tile, dict(seed=row['seed'], status=status,
        selected_rank=selected_rank, candidates=candidates,
        unselected_segmentation_masks_archived=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'results/unified_vision_archives_20261110_migration_v2_release/report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('Output must stay within ULA')
    report = json.loads(args.report.read_text())
    if report['run_status'] != 'finished':
        raise ValueError('Only render a completed archived evaluation')
    locks_path = args.report.parent/'prediction_locks.json'
    locks = json.loads(locks_path.read_text())
    rows = [row for row in report['rows'] if row['cohort']=='bottle' and row['sensor_available']
        and all(value['localization_verified'] for value in row['baseline'].values())
        and not all(value['localization_verified'] for value in row['unified'].values())]
    rows.sort(key=lambda row: row['seed'])
    if len(rows) != 16:
        raise ValueError(f'Expected the discussed 16 cases, found {len(rows)}')
    output.mkdir(parents=True, exist_ok=False)
    records, tiles = [], []
    preserved = {}
    for row in rows:
        sensor = Path(row['sensor_path'])
        prediction = Path(row['prediction_directory'])/'prediction.json'
        labels = Path(row['archive_directory'])/'evaluation_only_masks.npz'
        for path, expected in ((sensor,row['sensor_sha256']),
            (prediction,locks[str(prediction)]),(labels,row['independent_label_sha256'])):
            if digest(path) != expected:raise ValueError('Archived evidence changed: '+str(path))
            preserved[str(path)] = expected
        target = json.loads(prediction.read_text())['targets']['object']
        with np.load(sensor, allow_pickle=False) as data:
            rgb = data['rgb'][..., :3].copy()
            intrinsic, extrinsic = data['intrinsic'].copy(), data['world_to_camera'].copy()
        with np.load(labels, allow_pickle=False) as data:
            truth = data['bottle'].astype(bool)
        card, tile, record = render_case(rgb, truth, row, target, intrinsic, extrinsic)
        path = output/f"seed{row['seed']:05d}_original_and_candidates.png"
        card.save(path)
        original = output/f"seed{row['seed']:05d}_original.png"
        Image.fromarray(rgb).save(original)
        record.update(sensor_path=str(sensor),prediction_path=str(prediction),
            full_image=str(path),original_image=str(original))
        records.append(record);tiles.append(tile)
    sheets = []
    for start in range(0, len(tiles), 4):
        page = start//4+1
        sheet = Image.new('RGB', (1280, 1390), 'white')
        draw = ImageDraw.Draw(sheet)
        text(draw, (16, 4), f'药瓶：旧版通过、新版未通过的 16 场（{page}/4）', 28)
        text(draw, (16, 47), '绿＝位置接近；红＝其他候选；黄＝事后标签。框不是分割，绿也不是身份已核验。', 23)
        for index, tile in enumerate(tiles[start:start+4]):
            sheet.paste(tile, ((index%2)*640, 100+(index//2)*640))
        path = output/f'candidate_sheet_{page:02d}.png'
        sheet.save(path);sheets.append(str(path))
    unchanged = all(digest(path)==expected for path,expected in preserved.items())
    if not unchanged:raise ValueError('Rendering unexpectedly changed source evidence')
    summary = dict(source_report=str(args.report.resolve()), report_sha256=digest(args.report),
        cases=records,sheets=sheets,sources_unchanged=unchanged,
        model_inference_executed=False,robot_actions_executed=False,
        labels_used_for_posthoc_visualization_only=True,
        proximity_does_not_prove_segmentation_or_identity=True)
    (output/'gallery.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines = ['# 药瓶 16 场：原图与候选证据', '',
        '15 场拒绝输出，seed 7 误选。绿色仅表示候选可见三维中心与独立标签参考位置接近（≤2 cm），不是身份已确认。', '',
        '红色为其他候选，黄色为独立标签轮廓。框是保存的搜索框，不是分割边界；圆点是保存的三维候选中心投影。', '',
        '这些拒绝样本没有保存各个未选中候选的完整分割掩码，本图没有重新推理，也没有补画分割。', '']
    for record in records:
        lines.append(f"- seed {record['seed']}：[原图]({record['original_image']})｜[原图与候选对照]({record['full_image']})；{record['status']}")
    lines+=['','## 全部候选总览','']
    lines += [f'![候选总览 {i+1}]({path})' for i,path in enumerate(sheets)]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),seeds=[r['seed'] for r in records],
        sheets=sheets,sources_unchanged=unchanged),ensure_ascii=False))


if __name__ == '__main__':
    main()
