"""Posthoc per-frame evidence audit; no inference or controller changes."""
import json,hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3'
OUT=ROOT/'results/vision_ambiguity_frame_audit_20261112_v1'
TASKS=('open_microwave','open_laptop','rotate_qrcode','pick_dual_bottles')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def loadmask(p):
    with np.load(p) as f:return f['mask'].astype(bool)
def match(m,t):
    n=int((m&t).sum());precision=n/max(1,int(m.sum()))
    return dict(precision=precision,intersection=n,matched=n>=20 and precision>=.8)

def main():
    OUT.mkdir(exist_ok=False);(OUT/'frames').mkdir()
    cases=[];locks={};images=[]
    for task in TASKS:
        taskpath=SOURCE/'tasks'/task/'report.json';locks[str(taskpath)]=sha(taskpath)
        for row in read(taskpath)['rows']:
            if not row.get('evaluation'):continue
            predpath=Path(row['prediction_directory'])/'prediction.json';pred=read(predpath)
            keys=[k for k,t in pred['targets'].items() if any(isinstance(r,str) and r in
                ('ambiguous_CAD_consistent_candidates','ambiguous_joint_instance_assignment') for r in t['reasons'])]
            if not keys:continue
            folder=Path(row['directory']);sensor=folder/'rgbd.npz';label=folder/'evaluation_only_masks.npz'
            locks.update({str(p):sha(p) for p in (predpath,sensor,label,folder/'known_model_provenance.json')})
            with np.load(sensor) as f:rgb=f['rgb'][...,:3].copy()
            with np.load(label) as f:truth={k:f[k].astype(bool) for k in f.files}
            groups=defaultdict(list)
            for p in read(folder/'known_model_provenance.json'):groups[p['equivalence_group']].append(p['key'])
            eligible={k:vs for vs in groups.values() for k in vs}
            tiles=[]
            for key in keys:
                candidates=[];masks={}
                for g in pred['targets'][key]['candidates']:
                    for c in g['candidates']:
                        v=c.get('cad_verification') or {}
                        if not v.get('accepted'):continue
                        path=c['measured_mask_path'];locks[path]=sha(path);m=loadmask(path)
                        metrics={k:match(m,t) for k,t in truth.items()};matched=[k for k,vv in metrics.items() if vv['matched']]
                        item=dict(id=c['candidate_id'],cost=v['cost'],matches=matched,metrics=metrics,
                            eligible=bool(set(matched)&set(eligible[key])),pose=v['object_to_world'],mask_path=path)
                        candidates.append(item);masks[item['id']]=m
                candidates.sort(key=lambda x:x['cost']);pairs=[]
                for a,b in itertools.combinations(candidates,2):
                    pa,pb=np.asarray(a['pose']),np.asarray(b['pose']);ma,mb=masks[a['id']],masks[b['id']]
                    angle=float(np.degrees(np.arccos(np.clip((np.trace(pa[:3,:3].T@pb[:3,:3])-1)/2,-1,1))))
                    pairs.append(dict(a=a['id'],b=b['id'],mask_iou=float((ma&mb).sum()/max(1,(ma|mb).sum())),
                        translation_m=float(np.linalg.norm(pa[:3,3]-pb[:3,3])),rotation_deg=angle,
                        same_label=bool(set(a['matches'])&set(b['matches']))))
                good=[c for c in candidates if c['eligible']];bad=[c for c in candidates if not c['eligible']]
                identities={k for c in good for k in c['matches'] if k in eligible[key]}
                if not good:kind='no_precise_eligible_candidate'
                elif bad:kind='eligible_and_wrong_or_impure_candidates'
                elif len(identities)>1:kind='multiple_eligible_physical_instances'
                else:kind='same_eligible_instance_only'
                goodids={c['id'] for c in good}
                gp=[p for p in pairs if p['a'] in goodids and p['b'] in goodids and p['same_label']]
                pose_conflict=any(p['translation_m']>.03 or p['rotation_deg']>15 for p in gp)
                cases.append(dict(task=task,seed=row['seed'],target=key,eligible_labels=eligible[key],
                    category=kind,same_target_pose_spread_over_3cm_or_15deg=pose_conflict,
                    max_same_target_translation_m=max([p['translation_m'] for p in gp],default=None),
                    max_same_target_rotation_deg=max([p['rotation_deg'] for p in gp],default=None),
                    candidates=candidates,pairs=pairs,prediction=str(predpath)))
                # Every CAD-accepted candidate gets a panel, not only top-ranked.
                base=Image.fromarray(rgb).resize((480,360));dr=ImageDraw.Draw(base)
                dr.text((4,4),f'{task} seed={row["seed"]} {key}: RGB',fill='yellow');tiles.append(base)
                for c in candidates:
                    m=masks[c['id']];overlay=rgb.copy();overlay[m]=(overlay[m]*.45+np.array([0,140,0])).clip(0,255)
                    for k,tm in truth.items():
                        edge=tm&~(np.roll(tm,1,0)&np.roll(tm,-1,0)&np.roll(tm,1,1)&np.roll(tm,-1,1))
                        overlay[edge]=[255,220,0] if k in eligible[key] else [0,200,255]
                    im=Image.fromarray(overlay.astype('uint8')).resize((480,360));draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,479,42),fill='black')
                    draw.text((4,3),f'{key} {c["id"]} cost={c["cost"]:.3f}',fill='white')
                    draw.text((4,21),'label='+','.join(c['matches']) if c['matches'] else 'no precise label match',fill='white')
                    tiles.append(im)
            columns=3;sheet=Image.new('RGB',(480*columns,360*((len(tiles)+columns-1)//columns)),(35,35,35))
            for i,tile in enumerate(tiles):sheet.paste(tile,((i%columns)*480,(i//columns)*360))
            imagepath=OUT/'frames'/f'{task}_seed{row["seed"]:03d}.jpg';sheet.save(imagepath,quality=85);images.append(str(imagepath))
            print(task,row['seed'],keys,flush=True)
    summary={task:dict(Counter(c['category'] for c in cases if c['task']==task)) for task in TASKS}
    report=dict(scope='posthoc_saved_accepted_candidates_not_robot_actions',source=str(SOURCE),
        identity_rule='precision>=0.8 and overlap>=20pixels; equivalence groups respected',
        diagnostic_pose_flags_only='3cm/15deg are diagnostic, not new inference gates or action-equivalence proof',
        limitation='all saved accepted candidates; original final competing assignments not reconstructed; no true 6D pose labels',
        scenes=len(images),target_queries=len(cases),summary=summary,cases=cases,images=images,
        original_files_unchanged=all(sha(p)==h for p,h in locks.items()))
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'input_hashes.json').write_text(json.dumps(locks,indent=2)+'\n')
    lines=['# 逐帧歧义核验','',report['limitation'],'','|task|seed|target|category|max translation (cm)|max rotation (deg)|image|','|---|---:|---|---|---:|---:|---|']
    for c in cases:
        t=c['max_same_target_translation_m'];r=c['max_same_target_rotation_deg'];im=OUT/'frames'/f'{c["task"]}_seed{c["seed"]:03d}.jpg'
        lines.append(f'|{c["task"]}|{c["seed"]}|{c["target"]}|{c["category"]}|{t*100 if t is not None else "NA"}|{r if r is not None else "NA"}|[image]({im})|')
    (OUT/'index.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scenes=len(images),queries=len(cases),summary=summary),ensure_ascii=False))

if __name__=='__main__':main()
