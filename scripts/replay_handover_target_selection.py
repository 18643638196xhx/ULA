"""Check target selection on saved failures and successful control frames."""
import json
from pathlib import Path
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.handover.held_vision import cloud_from_frames
from adapters.handover.target_cloud import fit_target_cloud


def main():
    source=ROOT/'results/pi05_adaptive_full_50valid_seed40000_20261107_v1/report.json'
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/handover_target_selection_replay_20261107_v1')
    output=parser.parse_args().output
    output.mkdir(exist_ok=False)
    rows=json.loads(source.read_text())['episodes']
    report=dict(source=str(source),mode='saved_frames_not_closed_loop_success_rate',samples=[])
    for row in rows:
        directory=Path(row.get('placement_control',{}).get('vision_debug_dir','/nonexistent'))
        samples=[]
        if row.get('handover_success') and 'trustworthy measured' in row.get('error',''):
            index=max(int(p.name.split('_')[1]) for p in directory.glob('observation_*_head_camera.npz'))-1
            baseline=ROOT/f'results/handover_failed_cloud_visuals_20261107_v1/seed_{row["seed"]}_cloud.npz'
            samples=[('previous_failure',index,np.load(baseline)['prior'],None)]
        elif row.get('success'):
            files=sorted(directory.glob('observation_*_fit.npz'))
            for path in [files[0],files[-1]]:
                with np.load(path) as f:
                    samples.append(('successful_control',int(path.name.split('_')[1]),f['prior'],f['pose']))
        for kind,index,prior,old_pose in samples:
            frames={}
            for camera in ('head_camera','left_camera','right_camera'):
                with np.load(directory/f'observation_{index:03d}_{camera}.npz') as f:
                    frames[camera]={k:f[k] for k in f.files}
            points,_=cloud_from_frames(frames,prior)
            record=dict(seed=row['seed'],kind=kind,index=index,input_points=len(points))
            try:
                pose,quality,selected=fit_target_cloud(points,prior)
                record.update(accepted=True,selected_points=len(selected),quality=quality)
                if old_pose is not None:
                    record.update(previous_center_difference_m=float(np.linalg.norm(pose[:3,3]-old_pose[:3,3])),
                        previous_axis_difference_deg=float(np.degrees(np.arccos(np.clip(pose[:3,2]@old_pose[:3,2],-1,1)))))
                np.savez_compressed(output/f'seed_{row["seed"]}_{index:03d}.npz',
                    points=selected,candidate_points=points,pose=pose,prior=prior)
            except RuntimeError as error:
                record.update(accepted=False,error=str(error))
            report['samples'].append(record)
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(row['seed'],kind,index,'accepted',record['accepted'],
                  'selected',record.get('selected_points'), 'of',len(points),flush=True)
    report['summary']={kind:dict(total=sum(s['kind']==kind for s in report['samples']),
        accepted=sum(s['kind']==kind and s['accepted'] for s in report['samples']))
        for kind in ['previous_failure','successful_control']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
