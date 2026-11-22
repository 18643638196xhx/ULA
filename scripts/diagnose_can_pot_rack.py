"""Read-only saved prediction diagnosis; writes a new evidence directory."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3/tasks'
OUT=ROOT/'results/can_pot_rack_diagnosis_20261112_v2'

def main():
    OUT.mkdir(exist_ok=False);records=[]
    for task in ('move_can_pot','hanging_mug'):
        d=json.loads((SOURCE/task/'report.json').read_text())
        for row in d['rows']:
            if not row.get('evaluation'):continue
            failed=[(k,s) for k,s in row['evaluation']['targets'].items() if not s['localization_verified']]
            if not failed:continue
            case=Path(row['directory']);pred=json.loads((Path(row['prediction_directory'])/'prediction.json').read_text())
            with np.load(case/'rgbd.npz') as f:rgb=f['rgb'][...,:3].copy()
            with np.load(case/'evaluation_only_masks.npz') as f:truth={k:f[k].astype(bool) for k in f.files}
            for key,s in failed:
                t=pred['targets'][key];cs=[]
                for g in t['candidates']:
                    for c in g['candidates']:
                        with np.load(c['measured_mask_path']) as f:m=f['mask'].astype(bool)
                        overlaps={k:float((m&v).sum()/max(1,m.sum())) for k,v in truth.items()}
                        correct=bool(overlaps[key]>=.8 and (m&truth[key]).sum()>=20)
                        cv=c.get('cad_verification') or {}
                        cs.append(dict(id=c['candidate_id'],correct=correct,accepted=cv.get('accepted'),cost=cv.get('cost'),
                            overlaps=overlaps,position=c['position_world'],source=c['source'],score=c['semantic_score']))
                image=rgb.copy();selected=None
                if t['evidence']:
                    with np.load(t['evidence'][0]['measured_mask_path']) as f:selected=f['mask'].astype(bool)
                    image[selected]=(image[selected]*.5+np.array([0,125,0])).clip(0,255)
                mask=truth[key];edge=mask&~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
                image[edge]=[255,220,0]
                im=Image.fromarray(image.astype('uint8')).resize((960,720));draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,959,38),fill='black');draw.text((5,5),f'{task} seed={row["seed"]} {key} status={t["status"]} error={s.get("visible_surface_center_error_m")}',fill='white')
                path=OUT/f'{task}_seed{row["seed"]:03d}_{key}.png';im.save(path)
                item=dict(task=task,seed=row['seed'],key=key,score=s,reasons=[r for r in t['reasons'] if isinstance(r,str)],
                    geometry=t['geometry'],evidence=t['evidence'],candidates=cs,image=str(path),directory=str(case))
                records.append(item)
                if key in ('can','rack'):
                    accepted=[c for c in cs if c['accepted']];correct=[c for c in cs if c['correct']]
                    print(task,row['seed'],key,'output',s['output'],'right',s['identity_match'],
                        'correct_proposals',len(correct),'correct_accepted',sum(c['accepted'] for c in correct),
                        'selected',[(e.get('candidate_id'),e.get('source'),e.get('semantic_score')) for e in t['evidence']],
                        'best_accepted',sorted(accepted,key=lambda c:c['cost'])[:1],flush=True)
    (OUT/'report.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
