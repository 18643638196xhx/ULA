"""Validated declarative task rules for the shared ULA controllers."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SupportPlacementRule:
    task: str
    manipulated_object: str
    support_target: str
    grasp_detector: str
    stable_steps: int
    gripper_command_threshold: float
    trial_lift: bool
    candidate_budget: int
    safe_escape: str
    support_anchor: str
    target_anchor: str
    desired_orientation: str
    approach_direction: str
    distance_source: str
    support_type: str
    desired_quaternion_wxyz: object
    success: dict
    fixed_release_gap: object
    release_requirements: tuple
    retreat: str
    simulator_truth_for_control: bool
    simulator_truth_for_final_evaluation: bool
    source_path: str

    def to_dict(self):
        return asdict(self)


PLACEMENT_STAGES = [
    ("traditional", "approach_manipulated_object"),
    ("vla", "local_grasp"),
    ("traditional", "adaptive_support_placement"),
]
CONTAINER_STAGES = [
    ("traditional", "approach_manipulated_object"),
    ("vla", "local_grasp"),
    ("traditional", "container_placement"),
]
ARTICULATED_CONTAINER_STAGES = [
    ("traditional", "approach_manipulated_object"),
    ("vla", "local_object_grasp"),
    ("traditional", "approach_articulated_handle"),
    ("traditional", "engage_articulated_handle"),
    ("traditional", "open_prismatic_container"),
    ("traditional", "verified_object_lift"),
    ("traditional", "place_in_articulated_container"),
    ("traditional", "release_and_retreat"),
]
SUPPORTED_TARGET_ANCHORS = ("container_functional_point", "container_interior", "fixed_pose")
SUPPORTED_RELEASE_MODES = ("drop_into",)
SUPPORTED_CONTAINER_STAGES = CONTAINER_STAGES
PRESS_STAGES = [
    ("traditional", "approach_contact_point"),
    ("traditional", "actuate"),
]
SUPPORTED_PRESS_ACTIONS = ("press", "turn")


SHAKE_STAGES = [
    ("traditional", "approach_manipulated_object"),
    ("vla", "local_grasp"),
    ("traditional", "shake"),
]


@dataclass(frozen=True)
class ShakeRule:
    task: str
    manipulated_object: str
    axis: str
    amplitude_rad: float
    cycles: int
    lift_step_m: float
    pre_lift_m: float
    pre_quaternion_wxyz: tuple
    horizontal: bool
    success: dict
    simulator_truth_for_control: bool
    simulator_truth_for_final_evaluation: bool
    source_path: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ContainerRule:
    task: str
    manipulated_object: str
    support_target: str
    target_anchor: str
    release_mode: str
    dual_arm: dict
    targets: tuple
    success: dict
    simulator_truth_for_control: bool
    simulator_truth_for_final_evaluation: bool
    source_path: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArticulatedContainerRule:
    task: str
    manipulated_object: str
    articulated_container: str
    movable_link: str
    prismatic_joint: str
    contact_point_ids: tuple
    functional_point_id: int
    opening_source: str
    position_uncertainty_m: float
    gripper_clearance_m: float
    object_grasp: dict
    handle_grasp: dict
    success: dict
    simulator_truth_for_control: bool
    simulator_truth_for_final_evaluation: bool
    source_path: str

    def to_dict(self):
        return asdict(self)


def load_articulated_container_rule(path):
    import math
    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("Articulated-container rule requires schema_version: 2")
    stages = value.get("stages")
    actual = ([(item.get("controller"), item.get("operation")) for item in stages]
              if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
              else None)
    if actual != ARTICULATED_CONTAINER_STAGES:
        raise ValueError("Unsupported articulated-container control-stage sequence")
    entities = value.get("entities", {})
    articulation = value.get("articulation", {})
    opening = value.get("opening", {})
    truth = value.get("truth_policy", {})
    rule = ArticulatedContainerRule(
        task=value.get("task"),
        manipulated_object=entities.get("manipulated_object"),
        articulated_container=entities.get("articulated_container"),
        movable_link=articulation.get("movable_link"),
        prismatic_joint=articulation.get("prismatic_joint"),
        contact_point_ids=tuple(int(v) for v in articulation.get("contact_point_ids", ())),
        functional_point_id=int(articulation.get("functional_point_id", 0)),
        opening_source=opening.get("distance_source"),
        position_uncertainty_m=float(opening.get("position_uncertainty_m", .015)),
        gripper_clearance_m=float(opening.get("gripper_clearance_m", .04)),
        object_grasp=dict(value.get("object_grasp") or {}),
        handle_grasp=dict(value.get("handle_grasp") or {}),
        success=value.get("success", {}),
        simulator_truth_for_control=truth.get("simulator_truth_for_control"),
        simulator_truth_for_final_evaluation=truth.get("simulator_truth_for_final_evaluation"),
        source_path=str(path),
    )
    for label in ("task", "manipulated_object", "articulated_container",
                  "movable_link", "prismatic_joint"):
        if not isinstance(getattr(rule, label), str) or not getattr(rule, label):
            raise ValueError(f"Articulated-container rule needs {label}")
    if not rule.contact_point_ids or min(rule.contact_point_ids) < 0:
        raise ValueError("Articulated-container rule needs nonnegative contact point ids")
    if rule.functional_point_id < 0:
        raise ValueError("Articulated-container functional point id must be nonnegative")
    if rule.opening_source != "measured_object_extent_and_urdf_limit":
        raise ValueError("Drawer opening must be derived from measured object extent and URDF limit")
    if (not math.isfinite(rule.position_uncertainty_m)
            or not 0 < rule.position_uncertainty_m <= .05
            or not math.isfinite(rule.gripper_clearance_m)
            or not 0 < rule.gripper_clearance_m <= .12):
        raise ValueError("Invalid articulated-container clearance configuration")
    if rule.success and rule.success.get("source") != "env.check_success":
        raise ValueError("success.source must be env.check_success")
    if rule.simulator_truth_for_control is not False:
        raise ValueError("Simulator truth must be disabled for control")
    if rule.simulator_truth_for_final_evaluation is not True:
        raise ValueError("Final simulator evaluation must be declared explicitly")
    return rule


def load_container_rule(path):
    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Container rule requires schema_version: 1")
    stages = value.get("stages")
    actual = ([(item.get("controller"), item.get("operation")) for item in stages]
              if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
              else None)
    if actual != CONTAINER_STAGES:
        raise ValueError("Unsupported container control-stage sequence")
    entities = value.get("entities", {})
    placement = value.get("placement", {})
    targets = value.get("targets") or []
    if not isinstance(targets, list) or not targets:
        raise ValueError("Container rule needs a non-empty targets list")
    for target in targets:
        if not isinstance(target, dict) or not target.get("object") or not target.get("slot"):
            raise ValueError("Each target needs object and slot fields")
    truth = value.get("truth_policy", {})
    rule = ContainerRule(
        task=value.get("task"),
        manipulated_object=entities.get("manipulated_object"),
        support_target=entities.get("support_target"),
        target_anchor=placement.get("target_anchor"),
        release_mode=placement.get("release_mode"),
        dual_arm=dict(value.get("dual_arm") or {}),
        targets=tuple(dict(t) for t in targets),
        success=value.get("success", {}),
        simulator_truth_for_control=truth.get("simulator_truth_for_control"),
        simulator_truth_for_final_evaluation=truth.get("simulator_truth_for_final_evaluation"),
        source_path=str(path),
    )
    if not isinstance(rule.task, str) or not rule.task:
        raise ValueError("Container rule needs a task name")
    if not isinstance(rule.manipulated_object, str) or not rule.manipulated_object:
        raise ValueError("Container rule needs a manipulated object")
    if not isinstance(rule.support_target, str) or not rule.support_target:
        raise ValueError("Container rule needs a support target")
    if rule.target_anchor not in SUPPORTED_TARGET_ANCHORS:
        raise ValueError("Unsupported container target anchor")
    if rule.release_mode not in SUPPORTED_RELEASE_MODES:
        raise ValueError("Unsupported container release mode")
    if rule.success and rule.success.get("source") != "env.check_success":
        raise ValueError("success.source must be env.check_success")
    if rule.simulator_truth_for_control is not False:
        raise ValueError("Simulator truth must be disabled for control")
    if rule.simulator_truth_for_final_evaluation is not True:
        raise ValueError("Final simulator evaluation must be declared explicitly")
    return rule


@dataclass(frozen=True)
class PressRule:
    task: str
    category: str
    manipulated_object: str
    model_name: str
    model_id_attr: str
    action_type: str
    contact_point_id: int
    descend: str
    tolerance_m: float
    pre_grasp_m: float
    success: dict
    simulator_truth_for_control: bool
    simulator_truth_for_final_evaluation: bool
    source_path: str

    def to_dict(self):
        return asdict(self)


def load_press_rule(path):
    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Press rule requires schema_version: 1")
    stages = value.get("stages")
    actual = ([(item.get("controller"), item.get("operation")) for item in stages]
              if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
              else None)
    if actual != PRESS_STAGES:
        raise ValueError("Unsupported press control-stage sequence")
    entities = value.get("entities", {})
    action = value.get("action", {})
    truth = value.get("truth_policy", {})
    rule = PressRule(
        task=value.get("task"),
        category=value.get("category", "press"),
        manipulated_object=entities.get("manipulated_object"),
        model_name=entities.get("model_name"),
        model_id_attr=entities.get("model_id_attr"),
        action_type=action.get("type"),
        contact_point_id=int(action.get("contact_point_id", 0)),
        descend=action.get("descend"),
        tolerance_m=action.get("tolerance_m"),
        pre_grasp_m=action.get("pre_grasp_m"),
        success=value.get("success", {}),
        simulator_truth_for_control=truth.get("simulator_truth_for_control"),
        simulator_truth_for_final_evaluation=truth.get("simulator_truth_for_final_evaluation"),
        source_path=str(path),
    )
    if not isinstance(rule.task, str) or not rule.task:
        raise ValueError("Press rule needs a task name")
    if not isinstance(rule.manipulated_object, str) or not rule.manipulated_object:
        raise ValueError("Press rule needs a manipulated object")
    if rule.action_type not in SUPPORTED_PRESS_ACTIONS:
        raise ValueError("Unsupported press action type")
    if rule.action_type == "press" and rule.descend != "to_contact_point":
        raise ValueError("Press action needs descend: to_contact_point")
    if rule.success and rule.success.get("source") != "env.check_success":
        raise ValueError("success.source must be env.check_success")
    if rule.simulator_truth_for_control is not False:
        raise ValueError("Simulator truth must be disabled for control")
    if rule.simulator_truth_for_final_evaluation is not True:
        raise ValueError("Final simulator evaluation must be declared explicitly")
    return rule


def load_task_rule(path):
    """Dispatch a validated rule by its declared stage sequence."""
    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    stages = value.get("stages")
    actual = ([(item.get("controller"), item.get("operation")) for item in stages]
              if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
              else None)
    if actual == PLACEMENT_STAGES:
        return load_support_placement_rule(path)
    if actual == SHAKE_STAGES:
        return load_shake_rule(path)
    if actual == CONTAINER_STAGES:
        return load_container_rule(path)
    if actual == ARTICULATED_CONTAINER_STAGES:
        return load_articulated_container_rule(path)
    if actual == PRESS_STAGES:
        return load_press_rule(path)
    raise ValueError("Unsupported control-stage sequence in task rule")


def load_shake_rule(path):
    import math

    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Task rule requires schema_version: 1")
    stages = value.get("stages")
    actual = ([(item.get("controller"), item.get("operation")) for item in stages]
              if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
              else None)
    if actual != SHAKE_STAGES:
        raise ValueError("Unsupported shake control-stage sequence")
    entities = value.get("entities", {})
    shake = value.get("shake", {})
    truth = value.get("truth_policy", {})
    rule = ShakeRule(
        task=value.get("task"),
        manipulated_object=entities.get("manipulated_object"),
        axis=shake.get("axis", "y"),
        amplitude_rad=float(shake.get("amplitude_rad", 7 * math.pi / 8)),
        cycles=int(shake.get("cycles", 3)),
        lift_step_m=float(shake.get("lift_step_m", 0.05)),
        pre_lift_m=float(shake.get("pre_lift_m", 0.1)),
        pre_quaternion_wxyz=tuple(float(v) for v in shake.get("pre_quaternion_wxyz", (0.707, 0, 0, 0.707))),
        horizontal=bool(shake.get("horizontal", False)),
        success=value.get("success", {}),
        simulator_truth_for_control=truth.get("simulator_truth_for_control"),
        simulator_truth_for_final_evaluation=truth.get("simulator_truth_for_final_evaluation"),
        source_path=str(path),
    )
    if not isinstance(rule.task, str) or not rule.task:
        raise ValueError("Shake rule needs a task name")
    if not isinstance(rule.manipulated_object, str) or not rule.manipulated_object:
        raise ValueError("Shake rule needs a manipulated object")
    if rule.axis != "y":
        raise ValueError("Shake primitive currently supports the y axis only")
    if not math.isfinite(rule.amplitude_rad) or rule.amplitude_rad <= 0:
        raise ValueError("Shake amplitude must be positive")
    if rule.cycles < 1 or rule.lift_step_m <= 0 or rule.pre_lift_m < 0:
        raise ValueError("Shake cycle and lift values are invalid")
    if len(rule.pre_quaternion_wxyz) != 4:
        raise ValueError("Shake pre-quaternion needs four numbers")
    if rule.success and rule.success.get("source") != "env.check_success":
        raise ValueError("success.source must be env.check_success")
    if rule.simulator_truth_for_control is not False:
        raise ValueError("Simulator truth must be disabled for control")
    if rule.simulator_truth_for_final_evaluation is not True:
        raise ValueError("Final simulator evaluation must be declared explicitly")
    return rule


def load_support_placement_rule(path):
    import yaml

    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Support-placement rule requires schema_version: 1")
    entities = value.get("entities", {})
    handoff = value.get("grasp_handoff", {})
    placement = value.get("placement", {})
    release = value.get("release", {})
    truth = value.get("truth_policy", {})
    stages = value.get("stages")
    expected_stages = [
        ("traditional", "approach_manipulated_object"),
        ("vla", "local_grasp"),
        ("traditional", "adaptive_support_placement"),
    ]
    actual_stages = ([(item.get("controller"), item.get("operation")) for item in stages]
                     if isinstance(stages, list) and all(isinstance(item, dict) for item in stages)
                     else None)
    if actual_stages != expected_stages:
        raise ValueError("Unsupported control-stage sequence in support-placement rule")
    rule = SupportPlacementRule(
        task=value.get("task"),
        manipulated_object=entities.get("manipulated_object"),
        support_target=entities.get("support_target"),
        grasp_detector=handoff.get("detector"),
        stable_steps=handoff.get("stable_steps"),
        gripper_command_threshold=handoff.get("gripper_command_threshold"),
        trial_lift=handoff.get("trial_lift"),
        candidate_budget=placement.get("candidate_budget"),
        safe_escape=placement.get("safe_escape"),
        support_anchor=placement.get("support_anchor"),
        target_anchor=placement.get("target_anchor"),
        desired_orientation=placement.get("desired_orientation"),
        approach_direction=placement.get("approach_direction"),
        distance_source=placement.get("distance_source"),
        support_type=placement.get("support_type", "plane"),
        desired_quaternion_wxyz=placement.get("desired_quaternion_wxyz"),
        success=value.get("success", {}),
        fixed_release_gap=placement.get("fixed_release_gap"),
        release_requirements=tuple(release.get("require", ())),
        retreat=release.get("retreat"),
        simulator_truth_for_control=truth.get("simulator_truth_for_control"),
        simulator_truth_for_final_evaluation=truth.get("simulator_truth_for_final_evaluation"),
        source_path=str(path),
    )
    text_fields = (
        rule.task, rule.manipulated_object, rule.support_target, rule.grasp_detector,
        rule.safe_escape, rule.support_anchor, rule.target_anchor,
        rule.desired_orientation, rule.approach_direction, rule.distance_source,
        rule.retreat,
    )
    if any(not isinstance(item, str) or not item for item in text_fields):
        raise ValueError("Rule contains a missing or invalid string field")
    if (not isinstance(rule.stable_steps, int) or rule.stable_steps < 1
            or not isinstance(rule.candidate_budget, int) or rule.candidate_budget < 1
            or not isinstance(rule.gripper_command_threshold, (int, float))
            or not 0 < rule.gripper_command_threshold < 1):
        raise ValueError("Rule contains an invalid handoff or candidate budget")
    if rule.grasp_detector != "stable_closed_gripper_command" or rule.trial_lift is not False:
        raise ValueError("This controller requires stable closed-command handoff without trial lift")
    if rule.safe_escape != "measured_scene_corridor":
        raise ValueError("Unsupported safe-escape rule")
    if rule.support_anchor != "lowest_visible_object_layer_at_visual_table_height":
        raise ValueError("Unsupported support-anchor rule")
    if rule.target_anchor != "measured_support_center_at_median_surface_height":
        raise ValueError("Unsupported target-anchor rule")
    if rule.desired_orientation not in ("initial_world", "fixed_quaternion", "unconstrained"):
        raise ValueError("Unsupported desired-orientation rule")
    if rule.desired_orientation == "fixed_quaternion":
        quat = rule.desired_quaternion_wxyz
        if not (isinstance(quat, (list, tuple)) and len(quat) == 4):
            raise ValueError("fixed_quaternion requires desired_quaternion_wxyz with 4 numbers")
    if rule.success and rule.success.get("source") != "env.check_success":
        raise ValueError("success.source must be env.check_success")
    if rule.support_type not in ("plane",):
        raise ValueError("Unsupported support-surface type")
    if rule.approach_direction != "support_surface_normal":
        raise ValueError("Unsupported placement approach direction")
    if rule.fixed_release_gap is not None:
        raise ValueError("Adaptive rule must not contain a fixed release gap")
    allowed_release = {"support_contact_observed", "footprint_contained"}
    if ("support_contact_observed" not in set(rule.release_requirements)
            or not set(rule.release_requirements) <= allowed_release):
        raise ValueError("Release must always require support contact and only known extras")
    if rule.retreat != "selected_precontact":
        raise ValueError("Unsupported retreat rule")
    if rule.simulator_truth_for_control is not False:
        raise ValueError("Simulator truth must be disabled for control")
    if rule.simulator_truth_for_final_evaluation is not True:
        raise ValueError("Final simulator evaluation must be declared explicitly")
    return rule

