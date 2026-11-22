"""Read-only retained-candidate funnel, never an inference-time oracle."""
import json,hashlib,sys,time,argparse
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.unified_vision import RGBDFrame
from diagnostics.unified_vision_evaluation import visible_reference

TASKS=('place_cans_plasticbox','place_burger_fries','place_bread_skillet','place_can_basket',
       'press_stapler','move_stapler_pad','open_laptop','pick_dual_bottles','move_can_pot',
       'hanging_mug','place_object_scale','put_object_cabinet','place_object_stand',
       'rotate_qrcode','open_microwave','stamp_seal')
STAGES=('candidate_identity','candidate_position','CAD_identity','CAD_position','final_identity','final_position')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def identity(mask,truth):
    overlap=int((mask&truth).sum());precision=overlap/max(1,int(mask.sum()))
    return overlap>=20 and precision>=.8,precision,overlap
def can_assign(keys,edges):
    matrix=np.array([[int((a,b) in edges) for b in keys] for a in keys])
    left,right=linear_sum_assignment(-matrix)
    return int(matrix[left,right].sum())==len(keys)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT):p.error('Use a NEW ULA output')
    out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    source=ROOT/'results/unified_vision50_known_cad_randomized_50per_task_20261110_v2_first3'
    report=dict(status='running',source=str(source),scope='posthoc_800_initial_frames_no_retest_no_actions',
        tasks={},completed_scenes=0,target_queries=0,files_verified=0,
        caveats=['Only saved retained candidates; raw detector/SAM proposals discarded earlier are unavailable.',
                 'Candidate correctness uses posthoc labels, NEVER feeds selection or training.',
                 'Candidate position uses saved visible centre, not true CAD pose or hidden centre.',
                 'Per-stage scene counts enforce one-to-one equivalent-target reference assignment.',
                 'Candidate-position availability is not achievable policy success or guaranteed full-task success.'])
    records=[];hashes={}
    try:
        for task in TASKS:
            taskfile=source/'tasks'/task/'report.json';hashes[str(taskfile)]=sha(taskfile)
            old=read(taskfile);rows=[r for r in old['rows'] if r.get('evaluation')]
            if len(rows)!=50:raise ValueError(f'{task}: expected 50 scenes')
            counts=Counter();roles=defaultdict(Counter);causes=Counter();scene_rows=[]
            for row in rows:
                folder=Path(row['directory']);pred_dir=Path(row['prediction_directory'])
                # Verify ALL original locked prediction files, not just JSON.
                for path,h in read(folder/'prediction_locks.json').items():
                    if sha(path)!=h:raise RuntimeError('Prediction lock mismatch: '+path)
                    hashes[path]=h;report['files_verified']+=1
                sensor=folder/'rgbd.npz'
                if sha(sensor)!=row['sensor_sha256']:raise RuntimeError('Sensor hash mismatch')
                hashes[str(sensor)]=row['sensor_sha256']
                frame=RGBDFrame.from_npz(sensor,camera='head',capture_id='initial',depth_unit='mm')
                labels=folder/'evaluation_only_masks.npz';hashes[str(labels)]=sha(labels)
                with np.load(labels) as f:truth={k:f[k].astype(bool) for k in f.files}
                refs={k:visible_reference(frame,m)[0] for k,m in truth.items()}
                groups=defaultdict(list)
                provenance=folder/'known_model_provenance.json';hashes[str(provenance)]=sha(provenance)
                for item in read(provenance):groups[item['equivalence_group']].append(item['key'])
                allowed={k:g for g in groups.values() for k in g}
                pred=read(pred_dir/'prediction.json');cache={}
                def mask(path):
                    if path not in cache:
                        with np.load(path) as f:cache[path]=f['mask'].astype(bool)
                    return cache[path]
                stage_edges={s:set() for s in STAGES};case_records=[];panels=[]
                for key,target in pred['targets'].items():
                    candidates=[];seen=set()
                    for group in target['candidates']:
                        for c in group['candidates']:
                            path=c.get('measured_mask_path')
                            if path is None or path in seen:continue
                            seen.add(path);m=mask(path);checks={};cad=bool((c.get('cad_verification') or {}).get('accepted'))
                            position=c.get('position_world')
                            for other in allowed[key]:
                                good,precision,overlap=identity(m,truth[other]);error=None
                                if good and refs[other] is not None and position is not None:
                                    error=float(np.linalg.norm(np.asarray(position)-refs[other]))
                                located=good and error is not None and error<=.02
                                checks[other]=dict(identity=good,precision=precision,intersection=overlap,position_error_m=error,position=located)
                                if good:stage_edges['candidate_identity'].add((key,other))
                                if located:stage_edges['candidate_position'].add((key,other))
                                if cad and good:stage_edges['CAD_identity'].add((key,other))
                                if cad and located:stage_edges['CAD_position'].add((key,other))
                            candidates.append(dict(id=c.get('candidate_id'),cad_accepted=cad,
                                cad_cost=(c.get('cad_verification') or {}).get('cost'),mask_path=path,checks=checks))
                    score=row['evaluation']['targets'][key];matched=score['matched_reference']
                    # Reuse archived final evaluation EXACTLY; do not redefine
                    # saved final output or count an equivalent object twice.
                    if score['identity_match']:stage_edges['final_identity'].add((key,matched))
                    if score['localization_verified']:stage_edges['final_position'].add((key,matched))
                    flags={s:any(a==key for a,b in stage_edges[s]) for s in STAGES}
                    for s,v in flags.items():roles[key][s]+=v
                    roles[key]['total']+=1
                    if flags['final_position']:cause='final_pass'
                    elif not flags['candidate_identity']:cause='no_correct_retained_region'
                    elif not flags['candidate_position']:cause='correct_region_but_no_accurate_candidate_position'
                    elif not flags['CAD_position']:cause='accurate_candidate_lost_at_CAD_gate'
                    elif not flags['final_identity']:cause='accurate_CAD_candidate_available_but_not_selected'
                    else:cause='accurate_CAD_candidate_available_but_final_position_failed'
                    causes[cause]+=1
                    record=dict(task=task,seed=row['seed'],target=key,eligible=allowed[key],
                        target_visible=bool(truth[matched].any()),reference_available=refs[matched] is not None,
                        flags=flags,cause=cause,final_evaluation=score,candidates=candidates,
                        rejection_reasons=[r for r in target['reasons'] if isinstance(r,str)])
                    records.append(record);case_records.append(record)
                    if not flags['final_position']:
                        best=max(candidates,key=lambda c:max((10*v['position']+v['identity']+v['precision']*.01 for v in c['checks'].values()),default=-1),default=None)
                        selected=mask(target['evidence'][0]['measured_mask_path']) if target['evidence'] else np.zeros(frame.depth.shape,bool)
                        for title,m in [('best retained '+key,mask(best['mask_path']) if best else np.zeros_like(selected)),('final '+key,selected)]:
                            canvas=frame.rgb.copy();canvas[m]=(canvas[m]*.45+[0,130,0]).clip(0,255)
                            for other in allowed[key]:
                                tm=truth[other];edge=tm&~(np.roll(tm,1,0)&np.roll(tm,-1,0)&np.roll(tm,1,1)&np.roll(tm,-1,1));canvas[edge]=[255,220,0]
                            im=Image.fromarray(canvas.astype('uint8')).resize((480,360));dr=ImageDraw.Draw(im);dr.rectangle((0,0,479,32),fill='black');dr.text((4,4),title,fill='white');panels.append(im)
                scene_flags={s:all(can_assign(group,stage_edges[s]) for group in groups.values()) for s in STAGES}
                assert scene_flags['final_position']==row['evaluation']['all_targets_localized']
                counts.update(s for s,v in scene_flags.items() if v)
                scene_item=dict(seed=row['seed'],stages=scene_flags,failed_targets=[c['target'] for c in case_records if not c['flags']['final_position']])
                if panels:
                    sheet=Image.new('RGB',(960,360*((len(panels)+1)//2)),(30,30,30))
                    for i,im in enumerate(panels):sheet.paste(im,((i%2)*480,(i//2)*360))
                    image=out/'images'/f'{task}_seed{row["seed"]:03d}.jpg';sheet.save(image,quality=80);scene_item['image']=str(image)
                scene_rows.append(scene_item);report['completed_scenes']+=1;report['target_queries']+=len(case_records)
            report['tasks'][task]=dict(scenes=50,joint_counts={s:counts[s] for s in STAGES},
                per_target={k:dict(v) for k,v in roles.items()},target_causes=dict(causes),scenes_detail=scene_rows)
            write(out/'report.json',report);write(out/'target_records.json',records)
            print(task,dict(counts),dict(causes),flush=True)
        report.update(status='finished',input_files_unchanged=all(sha(f)==h for f,h in hashes.items()))
        write(out/'input_hashes.json',hashes)
    except BaseException as e:
        report.update(status='error',error=f'{type(e).__name__}: {e}');raise
    finally:
        write(out/'report.json',report);write(out/'target_records.json',records)
    lines=['# 800场分层诊断','','仅归档中保存的候选；不代表原始检测器召回率。所有阶段均要求同场全部目标一对一匹配。','',
        '|任务|正确区域候选|准确位置候选|CAD后正确区域|CAD后准确位置|最终区域正确|最终定位通过|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for task,t in report['tasks'].items():lines.append('|'+task+'|'+'|'.join(str(t['joint_counts'][s])+'/50' for s in STAGES)+'|')
    lines+=['','## 每场失败图：左侧为事后选出的最佳保留候选，右侧为最终输出','',
        '最佳候选用标签事后选择，仅诊断上限，不是算法实际选择或可宣称的成功率。绿色候选，黄色允许目标轮廓。','']
    for task,t in report['tasks'].items():
        lines+=['### '+task,'']
        for s in t['scenes_detail']:
            if s.get('image'):lines.append(f'- seed {s["seed"]}: {", ".join(s["failed_targets"])} [图片]({s["image"]})')
        lines.append('')
    (out/'index.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
