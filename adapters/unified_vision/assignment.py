"""Scene-level exclusive instance assignment, using only current observations.

Equivalent queries may exchange instance slots, not invent simulator identity.
Other nearly tied assignments remain ambiguous. Missing slots are explicit.
"""
from dataclasses import asdict
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .identity import selection_cost


def query_signature(query):
    value=asdict(query);value.pop('key')
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def same_instance(a,b,config):
    """Do not merge neighbouring objects just because their centres are close."""
    shared=False
    for left in a:
        for right in b:
            if left['frame'].camera!=right['frame'].camera:continue
            shared=True
            x,y=left['mask'].mask,right['mask'].mask
            if np.logical_and(x,y).sum()/max(1,np.logical_or(x,y).sum())>.5:return True
            lv=left.get('cad_verification') or {};rv=right.get('cad_verification') or {}
            lr=left.get('cad_render_depth_m');rr=right.get('cad_render_depth_m')
            if lv.get('accepted') and rv.get('accepted') and lr is not None and rr is not None:
                pa=np.asarray(lv['object_to_world']);pb=np.asarray(rv['object_to_world'])
                lm=np.isfinite(lr)&(lr>0);rm=np.isfinite(rr)&(rr>0)
                overlap=lm&rm
                if (np.linalg.norm(pa[:3,3]-pb[:3,3])<=config.cad.cross_view_translation_m and
                    overlap.sum()/max(1,(lm|rm).sum())>.5 and overlap.any() and
                    np.median(np.abs(lr[overlap]-rr[overlap]))<=config.cad.max_median_error_m):
                    return True
    if shared:return False
    # Without a common view, require overlapping measured 3-D surfaces.
    x=np.concatenate([r['points'] for r in a]);y=np.concatenate([r['points'] for r in b])
    tolerance=2*config.voxel_m
    return bool(np.mean(cKDTree(y).query(x)[0]<=tolerance)>.65 and
                np.mean(cKDTree(x).query(y)[0]<=tolerance)>.65)


def assign_instances(items,config):
    """items: [{query, options:[record-group], tracked:bool}].

    A Hungarian solve maximizes matched slots first, then measured quality.
    Near-tied alternatives are checked modulo equivalent-query permutations.
    """
    if not items:return {},dict(instance_count=0)
    instances=[];edges={}
    for row,item in enumerate(items):
        for group in item['options']:
            matches=[i for i,other in enumerate(instances)
                     if any(same_instance(group,g,config) for g in other)]
            if matches:
                column=matches[0]
                # Reject a bridging mask conservatively: do not split it into
                # two independently assignable objects.
                for old in reversed(matches[1:]):
                    instances[column].extend(instances.pop(old))
                    remapped={}
                    for (r,c),value in edges.items():
                        key=(r,column if c==old else c-1 if c>old else c)
                        previous=remapped.get(key)
                        if previous is None or (len(value[1]),-value[0])>(len(previous[1]),-previous[0]):
                            remapped[key]=value
                    edges=remapped
            else:column=len(instances);instances.append([])
            instances[column].append(group)
            query=item['query']
            cost=(max(selection_cost(r,config) for r in group)/config.cad.min_cost_gap
                  if query.cad is not None else
                  -float(np.mean([r.get('rank',r['mask'].score) for r in group]))/config.ambiguity_score_gap)
            edge=(row,column)
            if edge not in edges or (len(group),-cost)>(len(edges[edge][1]),-edges[edge][0]):
                edges[edge]=(cost,group)
    n,m=len(items),len(instances)
    spread=max([abs(value[0]) for value in edges.values()]+[1.])+1
    view_penalty=2*n*spread
    missing=(config.max_views*view_penalty+2*spread+1)*(n+1)
    forbidden=missing*(n+2)
    matrix=np.full((n,m+n),forbidden,float);matrix[:,m:]=missing
    for (r,c),(cost,group) in edges.items():matrix[r,c]=cost+(config.max_views-len(group))*view_penalty
    def solve(costs):
        rows,cols=linear_sum_assignment(costs)
        picked={int(r):int(c) for r,c in zip(rows,cols) if c<m and costs[r,c]<missing}
        return picked,float(sum(costs[r,c] for r,c in zip(rows,cols)))
    best,total=solve(matrix)
    signatures=[query_signature(item['query'])+(':'+item['query'].key if item.get('tracked') else '')
                for item in items]
    classes={s:[r for r,t in enumerate(signatures) if t==s] for s in signatures}
    def sets(picked):return {s:{picked[r] for r in rows if r in picked} for s,rows in classes.items()}
    original=sets(best);uncertain=set();alternatives=[]
    def describe(picked):
        return {items[r]['query'].key:dict(instance_id=f'instance_{c:03d}',
            candidate_ids=[v.get('candidate_id') for v in edges[r,c][1]],
            normalized_edge_cost=edges[r,c][0]) for r,c in picked.items()}
    for signature,columns in original.items():
        for column in columns:
            trial=matrix.copy();trial[classes[signature],column]=forbidden
            alternative,score=solve(trial)
            if len(alternative)!=len(best) or score-total>=1.-1e-9:continue
            changed=[s for s,values in sets(alternative).items() if values!=original[s]]
            uncertain.update(r for s in changed for r in classes[s])
            alternatives.append(dict(normalized_cost_gap=max(0.,score-total),
                affected_keys=[items[r]['query'].key for s in changed for r in classes[s]],
                **(dict(assignment=describe(alternative)) if config.audit_candidate_selection else {})))
    result={}
    for row,item in enumerate(items):
        signature=signatures[row];column=best.get(row)
        reason=('ambiguous_joint_instance_assignment' if row in uncertain else
                'no_distinct_instance_available' if column is None else None)
        result[item['query'].key]=dict(selected=None if reason else edges[row,column][1],
            reason=reason,instance_id=None if reason else f'instance_{column:03d}',
            interchangeable_keys=[items[r]['query'].key for r in classes[signature]],
            slot_is_semantic_identity=False,
            **(dict(best_assignment=describe(best),competing_assignments=alternatives)
               if config.audit_candidate_selection else {}))
    return result,dict(instance_count=m,query_count=n,matched_slots=len(best),
        ambiguous_keys=[items[r]['query'].key for r in sorted(uncertain)],
        alternatives=alternatives,uses_simulator_identity=False)
