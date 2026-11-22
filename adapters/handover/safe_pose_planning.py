"""Opt-in guarded planning; no changes to the vendor planner or other tasks."""
import numpy as np

from adapters.handover.rgbd_obstacles import ObstaclePlanningSession
from adapters.handover.mesh_path_guard import VisualMeshPathGuard


def measured_values(robot):
    entity=robot.right_entity
    return dict(zip((joint.get_name() for joint in entity.get_active_joints()),entity.get_qpos()))


def restore_collision_checks(motion):
    # The local CuRobo start-state helper can early-return after disabling its
    # rollout self cost. Restore locally; do not mutate the shared vendor code.
    motion.rollout_fn.robot_self_collision_constraint.enable_cost()
    motion.rollout_fn.primitive_collision_constraint.enable_cost()


class SafePosePlanningSession(ObstaclePlanningSession):
    def __enter__(self):
        super().__enter__()
        try:
            self.guard=VisualMeshPathGuard(self.scene,self.planner.active_joints_name,self.robot_config,
                ee_link=self.robot_config['robot_cfg']['kinematics']['ee_link'])
            self.result.update(independent_mesh_guard=True,carried_primitive_path_guard=True,
                search_spheres_are_not_execution_certificate=True,
                uses_physics_world=False,task_free_yaw_search=True)
            return self
        except BaseException:
            super().__exit__(None,None,None)
            raise

    def update(self,held_pose):
        value=super().update(held_pose)
        self.guard.bind(held_pose,measured_values(self.task.robot))
        return value

    def release(self):
        self.guard.release(measured_values(self.task.robot))

    def plan(self,*args,**kwargs):
        # The guard binds to the measured start. Hypothetical last_qpos plans
        # need explicit future full state; reject rather than certify the wrong
        # robot/gripper. The adaptive controller plans one actual segment at a time.
        if kwargs.get('last_qpos') is not None:
            return dict(status='Fail',planner_status='GUARD_REQUIRES_MEASURED_START')
        motion=self.planner.motion_gen
        restore_collision_checks(motion)
        from curobo.wrap.reacher.motion_gen import PoseCostMetric
        motion.update_pose_cost_metric(PoseCostMetric.reset_metric())
        try:
            plan=super().plan(*args,**kwargs)
        finally:
            restore_collision_checks(motion)
            motion.update_pose_cost_metric(PoseCostMetric.reset_metric())
        if plan['status']!='Success':return plan
        current=np.asarray(self.task.robot.get_right_arm_real_jointState(),float)[:6]
        try:
            guard=self.guard.check_path(current,plan['position'],measured_values(self.task.robot))
        except (RuntimeError,ValueError) as error:
            guard=dict(accepted=False,error=str(error),truth_used=False)
        self.result['plans'][-1]['independent_guard']=guard
        if not guard['accepted']:
            return dict(status='Fail',planner_status='INDEPENDENT_MESH_OR_CARRIED_OBJECT_REJECTED',
                independent_guard=guard,start_rejected=guard.get('start_rejected',False))
        return dict(plan,independent_guard=guard)
