#!/usr/bin/env python3
"""Read-only posthoc attribution for the frozen 50-task localization benchmark.

Writes NEW analysis artifacts only. Never changes predictions, labels, task
controllers, inference gates or the frozen benchmark report. No GPU required.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

STAGES={
    'no_geometrically_valid_current_candidate':'NO_CANDIDATE',
    'no_CAD_consistent_current_candidate':'CAD_REFUSED',
    'ambiguous_CAD_consistent_candidates':'AMBIGUOUS',
}
LABELS=dict(OK='通过',NO_CANDIDATE='未交出候选',CAD_REFUSED='CAD核验拒绝',
    AMBIGUOUS='多候选歧义',WRONG_REGION='输出区域不符',POSITION='区域对但位置未核验通过')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def identity(mask,truth):
    if mask is None:return False,0.,0
    mask=np.asarray(mask,bool);truth=np.asarray(truth,bool)
    count=int((mask&truth).sum());precision=count/max(1,int(mask.sum()))
    return bool(count>=20 and precision>=.8),float(precision),count


def stage(target,score):
    if score['localization_verified']:return 'OK'
    if score['output']:return 'WRONG_REGION' if not score['identity_match'] else 'POSITION'
    reason=next((r for r in target.get('reasons',[]) if isinstance(r,str)),None)
    if reason not in STAGES:raise ValueError('Unclassified failure: '+str(reason))
    return STAGES[reason]


def ambiguity_type(accepted,allowed):
    matching={key for candidate in accepted for key in candidate['matching_targets'] if key in allowed}
    wrong=sum(not any(key in allowed for key in candidate['matching_targets']) for candidate in accepted)
    if len(matching)>1:return 'multiple_eligible_instances'
    if matching and wrong:return 'eligible_target_and_distractor'
    if matching:return 'multiple_masks_of_same_eligible_instance'
    return 'no_precisely_segmented_eligible_instance'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();output=args.output.resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output==ROOT:
        parser.error('Use ULA source and a fresh ULA analysis output')
    output.mkdir(parents=True,exist_ok=False)
    root_report=json.loads((source/'report.json').read_text())
    if root_report['status']!='finished':raise ValueError('Only analyze a completed benchmark')
    fixed={source/'report.json':sha(source/'report.json'),source/'freeze.json':sha(source/'freeze.json')}
    report=dict(source=str(source),source_status='finished',scope='posthoc_attribution_only_no_retest',
        cause_counts_unit='target_queries_not_scenes',thresholds_changed=False,
        identity_rule='intersection>=20pixels_and_predicted_target_precision>=0.8',
        candidate_rule='report_both_assigned_reference_and_any_equivalent_reference',tasks={},global_stages={},
        root_causes_not_inferred_from_gate_names_alone=True)
    global_stages=Counter();global_details=Counter();failed_scene_target_counts=Counter()
    prediction_files=0;all_case_records=[];representatives=[]
    for index,name in enumerate(root_report['tasks'],1):
        task_file=source/'tasks'/name/'report.json';fixed[task_file]=sha(task_file)
        task=json.loads(task_file.read_text());task_counts=Counter();details=Counter();role_counts=defaultdict(Counter)
        role_details=defaultdict(Counter);confusions=Counter();records=[];duplicate_scenes=0
        visibility=defaultdict(Counter)
        examples={};first_success=None;equivalent_groups_with_same_model=0
        for row in task['rows']:
            if not row.get('evaluation'):continue
            directory=Path(row['directory']);pred_dir=Path(row['prediction_directory'])
            locks=json.loads((directory/'prediction_locks.json').read_text())
            for file,digest in locks.items():
                if sha(file)!=digest:raise RuntimeError('Frozen prediction modified: '+file)
                prediction_files+=1
            sensor=directory/'rgbd.npz'
            if sha(sensor)!=row['sensor_sha256']:raise RuntimeError('Sensor input modified')
            prediction=json.loads((pred_dir/'prediction.json').read_text())
            provenance=json.loads((directory/'known_model_provenance.json').read_text())
            groups=defaultdict(list)
            for item in provenance:groups[item['equivalence_group']].append(item['key'])
            allowed={key:values for values in groups.values() for key in values}
            equivalent_groups_with_same_model+=sum(len(v)>1 for v in groups.values())
            with np.load(directory/'evaluation_only_masks.npz',allow_pickle=False) as data:
                actual={key:data[key].copy() for key in data.files}
            for key,truth in actual.items():
                visibility[key]['scenes']+=1;visibility[key]['visible_scenes']+=bool(truth.any())
                visibility[key]['fewer_than_20_visible_pixels']+=int(truth.sum())<20
                visibility[key]['touches_image_border']+=bool(truth[0].any() or truth[-1].any() or truth[:,0].any() or truth[:,-1].any())
            mask_cache={}
            def mask(path):
                if path is None:return None
                if path not in mask_cache:
                    with np.load(path,allow_pickle=False) as raw:mask_cache[path]=raw['mask'].copy()
                return mask_cache[path]
            output_matches={};outputs={}
            for key,target in prediction['targets'].items():
                if not row['evaluation']['targets'][key]['output']:continue
                file=target['evidence'][0]['measured_mask_path'];outputs[key]=mask(file)
                output_matches[key]=[other for other,truth in actual.items() if identity(outputs[key],truth)[0]]
            duplicate=False
            for group in groups.values():
                found=Counter(other for key in group for other in output_matches.get(key,[]) if other in group)
                if any(count>1 for count in found.values()):duplicate=True
            duplicate_scenes+=duplicate
            failed_targets=sum(not value['localization_verified'] for value in row['evaluation']['targets'].values())
            if failed_targets:failed_scene_target_counts[failed_targets]+=1
            elif first_success is None:first_success=dict(seed=row['seed'],directory=str(directory))
            for key,target in prediction['targets'].items():
                score=row['evaluation']['targets'][key];code=stage(target,score)
                task_counts[code]+=1;role_counts[key][code]+=1
                item=dict(task=name,seed=row['seed'],target=key,stage=code,
                    matched_reference=score['matched_reference'],eligible_references=allowed[key],
                    directory=str(directory),prediction_directory=str(pred_dir),
                    visible_reference_available=bool(score.get('reference_position_world') is not None),
                    target_visible=score['mask_metrics']['target_visible'],
                    position_error_m=score.get('visible_surface_center_error_m'),
                    no_control_or_inference_feedback=True)
                if code=='OK':
                    records.append(item);continue
                candidates=[]
                for group in target.get('candidates',[]):
                    for c in group['candidates']:
                        metrics={other:identity(mask(c.get('measured_mask_path')),truth) for other,truth in actual.items()}
                        matches=[other for other,value in metrics.items() if value[0]]
                        verification=c.get('cad_verification') or {}
                        reasons=Counter(reason for h in verification.get('hypotheses',[]) for reason in h.get('reasons',[]))
                        candidates.append(dict(candidate_id=c.get('candidate_id'),matching_targets=matches,
                            cad_accepted=bool(verification.get('accepted')),cad_cost=verification.get('cost'),
                            box_xyxy=c['box_xyxy'],mask_touches_image_border=c['mask_touches_image_border'],
                            eligible_precision=max((metrics[other][1] for other in allowed[key]),default=0.),
                            eligible_intersection=max((metrics[other][2] for other in allowed[key]),default=0),
                            hypothesis_rejection_reasons=dict(reasons),source_mask=c.get('measured_mask_path')))
                item.update(candidates=candidates,output_matches=output_matches.get(key,[]))
                if code=='NO_CANDIDATE':
                    attempts=[r for r in target['reasons'] if isinstance(r,dict) and r.get('operation')=='semantic_mask_proposals']
                    counts=[r.get('count',0) for r in attempts]
                    category='no_mask_proposals_in_either_pass' if not any(counts) else 'mask_proposals_filtered_before_CAD'
                    details[category]+=1;role_details[key][category]+=1;item['attribution']=category
                elif code=='CAD_REFUSED':
                    assigned=any(score['matched_reference'] in c['matching_targets'] for c in candidates)
                    eligible=any(set(allowed[key])&set(c['matching_targets']) for c in candidates)
                    category='CAD_rejected_correct_eligible_foreground' if eligible else 'CAD_received_no_precise_eligible_foreground'
                    details[category]+=1;role_details[key][category]+=1;item['attribution']=category
                    details['CAD_rejected_correct_assigned_foreground' if assigned else 'CAD_no_precise_assigned_foreground']+=1
                    if not eligible:
                        secondary=('target_overlap_but_impure_or_too_small' if any(c['eligible_intersection']>=20 for c in candidates)
                                   else 'target_absent_or_less_than_20pixels_in_retained_CAD_candidates')
                        details[secondary]+=1;role_details[key][secondary]+=1
                elif code=='AMBIGUOUS':
                    accepted=[c for c in candidates if c['cad_accepted']]
                    category=ambiguity_type(accepted,allowed[key])
                    details[category]+=1;role_details[key][category]+=1;item['attribution']=category
                elif code=='WRONG_REGION':
                    elsewhere=[other for other in output_matches.get(key,[]) if other!=score['matched_reference']]
                    category='picked_another_requested_object' if elsewhere else 'distractor_or_impure_mask'
                    details[category]+=1;role_details[key][category]+=1;item['attribution']=category
                    for other in elsewhere:confusions[key+' -> '+other]+=1
                    if duplicate:details['wrong_output_in_equivalent_duplicate_scene']+=1
                else:
                    category='visible_range_centre_error_over_2cm' if score.get('visible_surface_center_error_m') is not None else 'position_reference_or_output_unavailable'
                    details[category]+=1;role_details[key][category]+=1;item['attribution']=category
                examples.setdefault((key,code),dict(seed=row['seed'],directory=str(directory),attribution=item.get('attribution')))
                records.append(item)
            all_case_records.extend(records[-len(prediction['targets']):])
        summary=task['summary']
        weak=min(summary['per_target'],key=lambda k:summary['per_target'][k]['localized']/summary['per_target'][k]['scenes'])
        failed_codes={key:value for key,value in role_counts[weak].items() if key!='OK'}
        dominant=max(failed_codes,key=failed_codes.get) if failed_codes else None
        representative=examples.get((weak,dominant),first_success)
        representatives.append(dict(task=name,joint_pass=summary['all_targets_localized'],weak_role=weak,**representative))
        report['tasks'][name]=dict(joint_pass=summary['all_targets_localized'],scene_count=summary['evaluated_scenes'],
            initialization_rejections=len(summary['initialization_rejected_seeds']),stages=dict(task_counts),
            per_role_stages={k:dict(v) for k,v in role_counts.items()},details=dict(details),
            per_role_details={k:dict(v) for k,v in role_details.items()},per_target=summary['per_target'],
            per_role_visibility={k:dict(v) for k,v in visibility.items()},
            cross_target_confusions=dict(confusions),equivalent_duplicate_output_scenes=duplicate_scenes,
            equivalent_groups_seen=equivalent_groups_with_same_model,
            failure_examples=[dict(target=k,stage=code,**example) for (k,code),example in examples.items()],
            representative=representative)
        global_stages.update(task_counts);global_details.update(details)
        print(f'[analysis] {index}/50 {name}: joint={summary["all_targets_localized"]}/50',flush=True)
    report.update(global_stages=dict(global_stages),global_details=dict(global_details),
        failed_scene_target_counts={str(k):v for k,v in failed_scene_target_counts.items()},
        verified_prediction_files=prediction_files,
        source_reports_unchanged=all(sha(p)==v for p,v in fixed.items()))
    if not report['source_reports_unchanged']:raise RuntimeError('Source reports changed during analysis')
    write_json(output/'analysis.json',report);write_json(output/'target_cases.json',all_case_records)
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for offset in range(0,len(representatives),12):
        height=min(4,(len(representatives)-offset+2)//3)*280
        sheet=Image.new('RGB',(960,height),'#eeeeee');draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(representatives[offset:offset+12]):
            x,y=index%3*320,index//3*280
            file=Path(row['directory'])/'initial_head.png'
            with Image.open(file) as im:sheet.paste(im.convert('RGB').resize((320,240)),(x,y+40))
            draw.text((x+3,y+2),row['task'],font=font,fill='black')
            draw.text((x+3,y+20),f"joint {row['joint_pass']}/50 | seed {row['seed']} | {row['weak_role']}",font=font,fill='black')
        sheet.save(output/f'representative_scenes_{offset//12+1:02d}.png')
    lines=['# 50 任务定位：逐任务统计证据','',
        '只读分析已冻结的预测和同帧独立标签，没有修改原结果或重测。本表失败数以目标查询计数，不是场景数；一场可有多个目标失败。',
        '', '| 任务 | 联合通过 | 未交出候选 | CAD拒绝 | 多候选歧义 | 输出区域不符 | 区域对但位置未通过 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,t in report['tasks'].items():
        values=[t['stages'].get(key,0) for key in ('NO_CANDIDATE','CAD_REFUSED','AMBIGUOUS','WRONG_REGION','POSITION')]
        lines.append('| '+name+' | '+str(t['joint_pass'])+'/50 | '+' | '.join(map(str,values))+' |')
    for name,t in report['tasks'].items():
        lines+=['',f'## {name}：{t["joint_pass"]}/50','',
            '分目标定位通过：'+'；'.join(f'{key} {v["localized"]}/{v["scenes"]}' for key,v in t['per_target'].items())+'。',
            '', '目标失败分布：', '', '| 目标 | 未交候选 | CAD拒绝 | 歧义 | 输出错误 | 位置未通过 |', '|---|---:|---:|---:|---:|---:|']
        for key,counts in t['per_role_stages'].items():
            lines.append('| '+key+' | '+' | '.join(str(counts.get(code,0)) for code in ('NO_CANDIDATE','CAD_REFUSED','AMBIGUOUS','WRONG_REGION','POSITION'))+' |')
        lines+=['','事后细分（CAD候选按所有等价目标核对）：`'+json.dumps(t['details'],ensure_ascii=False)+'`。',
            '', '其他任务目标被错选的记录：`'+json.dumps(t['cross_target_confusions'],ensure_ascii=False)+'`。',
            '', f'等价多实例重复输出场景：{t["equivalent_duplicate_output_scenes"]}。',
            '', '代表场景：[原图]('+str(Path(t['representative']['directory'])/'initial_head.png')+')、[原评分叠加图]('+str(Path(t['representative']['directory'])/'audit.png')+')。']
    (output/'evidence.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),stages=report['global_stages'],details=report['global_details'],verified_prediction_files=prediction_files),ensure_ascii=False))


if __name__=='__main__':main()
