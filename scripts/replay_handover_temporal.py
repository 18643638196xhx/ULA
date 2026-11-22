"""Replay robot motion and saved RGB-D; object truth is post-update audit only."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from adapters.handover.temporal_vision import TemporalHeldTrack
from adapters.handover.held_vision import cloud_from_frames
from adapters.handover.target_cloud import fit_target_cloud


def main():
    source=ROOT/'results/pi05_component_full_pilot5_seed40026_20261107_v1/report.json'
    output=ROOT/'results/handover_temporal_replay_20261107_v1'
    output.mkdir(exist_ok=False)
    report=dict(source=str(source),mode='old_motion_sequence_not_new_closed_loop_score',episodes=[])
    for episode in json.loads(source.read_text())['episodes']:
        directory=Path(episode['placement_control']['vision_debug_dir'])
        track=TemporalHeldTrack();row=dict(seed=episode['seed'],frames=[])
        report['episodes'].append(row)
        first=np.load(directory/'observation_000_fit.npz')
        # The original prior was formed solely as robot FK @ attachment.
        first_hand=first['prior']@np.linalg.inv(np.asarray(episode['right_attachment']['right_ee_to_block']))
        for observation in episode['placement_control']['observations']:
            iteration=observation['iteration']
            accepted=[a for a in observation['attempts'] if a['accepted']]
            index=accepted[-1]['quality']['observation_index'] if accepted else max(
                int(p.name.split('_')[1]) for p in directory.glob('observation_*_head_camera.npz'))-1
            current=dict(iteration=iteration,index=index)
            row['frames'].append(current)
            frames={}
            for camera in ('head_camera','left_camera','right_camera'):
                with np.load(directory/f'observation_{index:03d}_{camera}.npz') as f:
                    frames[camera]={k:f[k] for k in f.files}
            try:
                if iteration==0:
                    points,_=cloud_from_frames(frames,first['prior'])
                    pose,quality,selected=fit_target_cloud(points,first['prior'])
                    pose,history=track.initialize(first_hand,pose,quality)
                else:
                    previous=next(x for x in episode['placement_observations_evaluation_only']
                        if x['stage'].startswith(f'feedback_{iteration-1:02d}_'))
                    # Allowed FK input; the actor pose in the same audit dict
                    # is deliberately not passed into either track function.
                    hand=np.asarray(previous['right_ee_world'])
                    predicted=track.predict(hand)
                    points,_=cloud_from_frames(frames,predicted)
                    selected,selection=track.select(points)
                    pose,quality=track.update(selected)
                    history=quality['temporal_tracking']
                    # Independent ground-truth check AFTER state update only.
                    actual=np.asarray(previous['actual_block_world'])
                    current['evaluation_only_center_error_m']=float(np.linalg.norm(pose[:3,3]-actual[:3,3]))
                    current['evaluation_only_bottom_error_m']=float(np.linalg.norm(
                        (pose[:3,3]-.1*pose[:3,2])-(actual[:3,3]-.1*actual[:3,2])))
                current.update(accepted=True,history=history,selected_points=len(selected))
            except RuntimeError as error:
                current.update(accepted=False,error=str(error));break
        row['all_saved_frames_accepted']=all(x['accepted'] for x in row['frames'])
        print(episode['seed'],'accepted',sum(x['accepted'] for x in row['frames']),
              'of',len(row['frames']),'last',str(row['frames'][-1])[:500],flush=True)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
