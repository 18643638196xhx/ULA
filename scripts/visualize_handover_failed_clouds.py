"""Read-only replay of saved failed RGB-D fits; never runs robot control."""
import ast
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tra_pipe/RoboTwin')]
from adapters.handover import held_vision as hv
from adapters.cup_target_focus import world_points_image
from adapters.handover.initial_vision import color_masks
from scipy.spatial.transform import Rotation

OUT = ROOT / 'results/handover_failed_cloud_visuals_20261107_v1'
REPORT = ROOT / 'results/pi05_adaptive_full_50valid_seed40000_20261107_v1/report.json'
font_manager.fontManager.addfont('/usr/share/fonts/SourceHanSansSC-Regular.otf')
plt.rcParams.update({'font.family': 'Source Han Sans SC', 'axes.unicode_minus': False,
                     'font.size': 10, 'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2476b8', '#e16a16', '#177b54'


def ee_matrix(pose):
    m = np.eye(4)
    m[:3, 3] = pose[:3]
    m[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    return m


def replay(row):
    directory = Path(row['placement_control']['vision_debug_dir'])
    index = max(int(p.name.split('_')[1]) for p in directory.glob('observation_*_head_camera.npz')) - 1
    observations = row['placement_control']['observations']
    binding = observations[-2]['right_ee_to_block'] if len(observations) > 1 else row['right_attachment']['right_ee_to_block']
    # On these failures no physics advances between the failed frame and final
    # robot diagnostic. Use robot FK only; never use actual_block_world.
    hand = ee_matrix(row['final_evaluation']['diagnostics_after_control_only']['actual_right_ee_pose'])
    prior = hand @ np.asarray(binding)
    frames = {}
    masks = {}
    for camera in ('head_camera', 'left_camera', 'right_camera'):
        with np.load(directory / f'observation_{index:03d}_{camera}.npz') as saved:
            frame = {k: saved[k] for k in saved.files}
        frames[camera] = frame
        world, valid = world_points_image(frame['depth'], frame['intrinsic'], frame['world_to_camera'])
        mask = color_masks(frame['rgb'])[0] & valid & (np.linalg.norm(world-prior[:3, 3], axis=2) < .23)
        interior = binary_erosion(mask)
        masks[camera] = interior if interior.sum() >= 60 else mask
    points, views = hv.cloud_from_frames(frames, prior)
    _, indices = np.unique(np.floor(points/.002).astype(np.int64), axis=0, return_index=True)
    cloud = points[np.sort(indices)]
    if len(cloud) > 2500:
        cloud = cloud[np.linspace(0, len(cloud)-1, 2500).astype(int)]
    original = hv.surface_distance
    fits = []
    def capture(p, pose):
        errors = original(p, pose)
        if len(p) == len(cloud):
            fits.append((float(np.sqrt(np.mean(errors**2))), pose.copy(), errors.copy()))
        return errors
    hv.surface_distance = capture
    try:
        hv.fit_cuboid(points, prior)
    except RuntimeError:
        pass
    finally:
        hv.surface_distance = original
    rmse, pose, errors = min(fits, key=lambda item: item[0])
    logged = ast.literal_eval(observations[-1]['attempts'][0]['error'].split('credible pose: ', 1)[1])
    expected = min(logged['attempts'], key=lambda a: a['rmse_m'])['rmse_m']
    if abs(rmse-expected) > 1e-6:
        raise RuntimeError(f'Offline replay mismatch seed {row["seed"]}: {rmse} vs {expected}')
    np.savez_compressed(OUT/f'seed_{row["seed"]}_cloud.npz', points=cloud, best_rejected_pose=pose,
                        signed_surface_errors_m=errors, prior=prior)
    return dict(seed=row['seed'], cloud=cloud, pose=pose, errors=errors, rmse=rmse,
                frames=frames, masks=masks, views=views, frame_index=index,
                iteration=observations[-1]['iteration'], input_dir=str(directory))


def cloud_plot(ax, data, compact=False):
    center = data['pose'][:3, 3]
    xyz = (data['cloud']-center)*100
    near = np.abs(data['errors']) < .006
    ax.scatter(*xyz[near].T, s=2 if compact else 5, c=BLUE, alpha=.5, depthshade=False,
               label=f'距表面 <6 mm：{near.sum()}点')
    ax.scatter(*xyz[~near].T, s=14 if compact else 28, c=ORANGE, marker='x', alpha=.95,
               depthshade=False, label=f'距表面 ≥6 mm：{(~near).sum()}点')
    signs = np.asarray(list(itertools.product((-1, 1), repeat=3)))
    corners = signs*hv.HALF
    corners = corners @ data['pose'][:3, :3].T*100
    for i, a in enumerate(corners):
        for j in range(i+1, len(corners)):
            # Edges are defined in unrotated box coordinates.
            if np.count_nonzero(signs[i] != signs[j]) == 1:
                ax.plot(*np.vstack((a, corners[j])).T, color=GREEN, lw=1.5)
    all_points = np.vstack((xyz, corners))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    mid = (low+high)/2
    half = max(high-low)*.56
    ax.set(xlim=(mid[0]-half, mid[0]+half), ylim=(mid[1]-half, mid[1]+half),
           zlim=(mid[2]-half, mid[2]+half), xlabel='X (cm)', ylabel='Y (cm)', zlabel='Z (cm)')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-62)
    ax.tick_params(labelsize=8)
    ax.set_title(f'seed {data["seed"]} · 误差 {data["rmse"]*1000:.2f} mm\n'
                 f'{near.mean()*100:.2f}% 的点距拟合表面不足 6 mm', fontsize=10 if compact else 13)
    if not compact:
        ax.legend(loc='upper left', bbox_to_anchor=(-.02, .01), fontsize=10)


def detail(data):
    fig = plt.figure(figsize=(12, 8.2))
    grid = fig.add_gridspec(3, 2, width_ratios=(1, 2.05), wspace=.12, hspace=.25)
    for i, (camera, label) in enumerate((('head_camera', '头部'), ('left_camera', '左腕'), ('right_camera', '右腕'))):
        ax = fig.add_subplot(grid[i, 0])
        ax.imshow(data['frames'][camera]['rgb'])
        mask = data['masks'][camera]
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0, 1, 1, .65]
        ax.imshow(overlay)
        suffix = '（不足15点，不合并）' if mask.sum() < 15 else ''
        ax.set_title(f'{label}相机：筛出 {mask.sum()} 个像素{suffix}', fontsize=10)
        ax.axis('off')
    ax = fig.add_subplot(grid[:, 1], projection='3d')
    cloud_plot(ax, data)
    fig.suptitle('失败帧 RGB-D 点云与被拒绝的最佳拟合', fontsize=16, y=.98)
    fig.text(.5, .025, '左：青色覆盖为当前筛选区域。右：蓝点贴近拟合表面，橙色 × 偏离较大；绿框为 6×6×20 cm 长方体。\n'
             '点的颜色仅表示拟合残差，不代表真实物体身份；坐标以拟合中心为零点，轴向与世界坐标一致。',
             ha='center', va='bottom', fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.15, left=.025, right=.97)
    path = OUT/f'seed_{data["seed"]}_detail.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    data = []
    for row in report['episodes']:
        if row.get('handover_success') and 'trustworthy measured' in row.get('error', ''):
            value = replay(row)
            detail(value)
            data.append(value)
            print('rendered', value['seed'], flush=True)
    fig = plt.figure(figsize=(14, 22))
    for index, item in enumerate(data):
        cloud_plot(fig.add_subplot(5, 2, index+1, projection='3d'), item, compact=True)
    fig.suptitle('全部 10 次视觉拟合拒绝：蓝点贴近表面，橙色 × 为偏离点，绿框为最佳拟合', fontsize=16)
    fig.subplots_adjust(top=.94, bottom=.03, left=.02, right=.98, wspace=.04, hspace=.45)
    fig.savefig(OUT/'all_10_failed_clouds.png', dpi=150)
    plt.close(fig)
    metadata = [{k:v for k,v in item.items() if k not in ('cloud','pose','errors','frames','masks')}
                for item in data]
    (OUT/'replay_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
