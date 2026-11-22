"""Task-free yaw search, not seed-specific wrist angles or a learned policy."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from traditional_place_empty_cup.geometry import matrix_to_pose_wxyz
from adapters.handover.place_control import settle_endpoint,slower_trajectory


def rigid(value):
    value=np.asarray(value,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all() or
        not np.allclose(value[3],[0,0,0,1]) or
        not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-6) or
        not np.isclose(np.linalg.det(value[:3,:3]),1.,atol=1e-6)):
        raise ValueError('A finite rigid transform is required')
    return value


def support_yaw_target(target,binding,angle,*,anchor,normal):
    """Rotate the desired object about its support anchor, not about the TCP."""
    target,binding=rigid(target),rigid(binding)
    normal=np.asarray(normal,dtype=float)
    anchor=np.asarray(anchor,dtype=float)
    if (normal.shape!=(3,) or anchor.shape!=(3,) or not np.isfinite(anchor).all() or
        not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9 or not np.isfinite(angle)):
        raise ValueError('Finite support normal, anchor and angle required')
    desired=target@binding
    support=desired[:3,3]+desired[:3,:3]@anchor
    desired[:3,:3]=Rotation.from_rotvec(normal/np.linalg.norm(normal)*angle).as_matrix()@desired[:3,:3]
    desired[:3,3]=support-desired[:3,:3]@anchor
    return desired@np.linalg.inv(binding)


@dataclass(frozen=True)
class YawSearchBudget:
    step_deg: float=15.
    max_abs_deg: float=180.
    max_plans: int=26

    def angles(self):
        if (not np.isfinite(self.step_deg) or not np.isfinite(self.max_abs_deg) or
            not 0<self.step_deg<=self.max_abs_deg<=180 or self.max_plans<1):
            raise ValueError('A finite bounded yaw discretization is required')
        values=[0.]
        for angle in np.arange(self.step_deg,self.max_abs_deg+1e-9,self.step_deg):
            values.append(float(angle))
            if angle<180-1e-9:values.append(float(-angle))
        return tuple(values)


class SupportYawExecutor:
    """Try old goal first, then bounded free-yaw goals; execute ONLY a guarded plan.

    The lowest-magnitude yaw that passes the independent guard wins; no fixed
    per-seed preferred angle. Existing arrival/release gates are not modified.
    """
    def __init__(self,budget=YawSearchBudget()):
        self.budget=budget
        self.budget.angles()

    def __call__(self,task,target,audit,*,rotation=False,binding,visual,normal,anchor,position_usable):
        rigid(visual)
        audit.update(motion='task_free_yaw_guarded_translation',yaw_search=[],
            arrival_gates_unchanged=True,support_anchor_preserved=True,
            yaw_allowed=bool(position_usable),settling={})
        candidates=[(0.,not rotation)]
        # Separate endpoint freedom from path orientation freedom. The second
        # attempt has exactly the old endpoint, without holding old wrist yaw.
        if not rotation:candidates.append((0.,False))
        if position_usable:
            candidates.extend((yaw,False) for yaw in self.budget.angles() if yaw)
        chosen=None
        for yaw,hold in candidates[:self.budget.max_plans]:
            candidate=support_yaw_target(target,binding,np.radians(yaw),anchor=anchor,normal=normal)
            kwargs=dict(constraint_pose=[1,1,1,0,0,0]) if hold else {}
            plan=task.robot.right_plan_path(matrix_to_pose_wxyz(candidate),**kwargs)
            entry=dict(yaw_deg=yaw,hold_path_orientation=hold,status=plan['status'],
                planner_status=plan.get('planner_status'),target_ee_pose=matrix_to_pose_wxyz(candidate).tolist(),
                independent_guard=plan.get('independent_guard'))
            audit['yaw_search'].append(entry)
            if plan['status']=='Success':
                # Require the isolated planning session, never an unguarded
                # ordinary CuRobo response in an accidentally miswired entry.
                if not plan.get('independent_guard',{}).get('accepted'):
                    raise RuntimeError('Yaw search requires the independent robot/object geometry guard')
                chosen=plan,candidate,yaw,hold
                break
            # Every candidate shares the same physical start and map. Turning
            # the endpoint cannot rescue an invalid start; do not waste 26 plans.
            if plan.get('start_rejected') or 'INVALID_START' in str(plan.get('planner_status')):
                break
        if chosen is None:
            raise RuntimeError('No independently guarded path within task-free yaw budget; no release')
        plan,candidate,yaw,hold=chosen
        audit.update(status='Success',chosen_yaw_deg=yaw,hold_path_orientation=hold,
            target_ee_pose=matrix_to_pose_wxyz(candidate).tolist(),
            independent_guard=plan['independent_guard'])
        moved=bool(task.take_dense_action(dict(left_arm=None,right_arm=slower_trajectory(plan,4.),
            left_gripper=None,right_gripper=None),save_freq=None))
        settle_endpoint(task,plan,candidate,audit['settling'])
        if not moved or not audit['settling']['ready']:
            raise RuntimeError('Guarded placement segment did not reach measured robot target; no release')
