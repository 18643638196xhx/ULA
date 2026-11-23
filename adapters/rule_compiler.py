"""Compile offline ``ula_demo_primitives_v1`` YAML rules into an executable runtime plan."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from adapters.primitive_geometry import _transform

SUPPORTED_SCHEMA = "ula_demo_primitives_v1"
SUPPORTED_PRIMITIVES = ("move", "grasp", "release", "handover", "place", "rotate", "press", "strike", "open")


@dataclass(frozen=True)
class CompiledRelation:
    name: str
    reference: str
    controlled: str
    frame: int
    T_reference_controlled: list[list[float]]
    source: str = "demonstration"


@dataclass(frozen=True)
class CompiledStage:
    stage_id: str
    primitive: str
    controller: str
    executor: str
    role: str
    arm: str | None
    donor_arm: str | None
    receiver_arm: str | None
    object_name: str | None
    reference_name: str | None
    demo_frames: tuple[int, int]
    depends_on: tuple[str, ...]
    relations: dict[str, CompiledRelation] = field(default_factory=dict)
    reusable_object_goal: list[list[float]] | None = None
    gripper_open_value: float = 1.0
    motion_delta_xyz_m: tuple[float, float, float] | None = None
    relative_rotation: list[list[float]] | None = None
    relative_rotation_role: str | None = None
    requires_runtime_tool_frame_reconciliation: bool = True
    child_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledRulePlan:
    schema: str
    task: str
    description: str
    active_arms: tuple[str, ...]
    bindings: dict[str, Any]
    stages: tuple[CompiledStage, ...]
    source_path: str | None = None
    warnings: tuple[str, ...] = ()

    def stage_map(self) -> dict[str, CompiledStage]:
        return {stage.stage_id: stage for stage in self.stages}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task": self.task,
            "description": self.description,
            "active_arms": list(self.active_arms),
            "bindings": dict(self.bindings),
            "source_path": self.source_path,
            "warnings": list(self.warnings),
            "stages": [s.to_dict() for s in self.stages],
        }


def _extract_relations(stage_dict: Mapping[str, Any]) -> dict[str, CompiledRelation]:
    rels: dict[str, CompiledRelation] = {}
    sid = stage_dict.get("id")
    geom = stage_dict.get("geometry") or {}
    for rec in geom.get("demonstrated_relations") or []:
        rname = rec["name"]
        mat = _transform(rec["T_reference_controlled"], f"{sid}.{rname}")
        rels[str(rname)] = CompiledRelation(
            name=str(rname),
            reference=str(rec["reference"]),
            controlled=str(rec["controlled"]),
            frame=int(rec.get("frame", -1)),
            T_reference_controlled=mat.tolist(),
            source=str(rec.get("source", "demonstration")),
        )
    return rels


def _extract_object_goal(stage_dict: Mapping[str, Any], bindings: Mapping[str, Any]) -> list[list[float]] | None:
    sid = stage_dict.get("id")
    raw_goal = stage_dict.get("object_goal")
    if isinstance(raw_goal, Mapping):
        for key in ("T_target_object", "T_reference_controlled", "matrix"):
            if raw_goal.get(key) is not None:
                return _transform(raw_goal[key], f"{sid}.object_goal.{key}").tolist()
        for k, v in raw_goal.items():
            if k.startswith("T_") and v is not None:
                return _transform(v, f"{sid}.object_goal.{k}").tolist()
    elif raw_goal is not None:
        return _transform(raw_goal, f"{sid}.object_goal").tolist()

    geom = stage_dict.get("geometry") or {}
    if geom.get("reusable_object_goal") is not None:
        return _transform(geom["reusable_object_goal"], f"{sid}.reusable_object_goal").tolist()

    for child in stage_dict.get("children") or []:
        cid = child.get("id")
        cgeom = child.get("geometry") or {}
        if cgeom.get("reusable_object_goal") is not None:
            return _transform(cgeom["reusable_object_goal"], f"{cid}.reusable_object_goal").tolist()
        for rec in cgeom.get("demonstrated_relations") or []:
            rname = rec.get("name")
            if rname in ("goal", "held_transport", "prerelease_object_goal"):
                return _transform(rec["T_reference_controlled"], f"{cid}.{rname}").tolist()
    return None


def _default_executor(primitive: str, controller: str) -> str:
    if primitive == "move":
        return "curobo"
    if primitive == "grasp":
        return "local_policy"
    if primitive == "release":
        return "gripper"
    if primitive == "handover":
        return "handover_state_machine"
    if primitive == "place":
        return "curobo_place"
    if primitive == "rotate":
        return "constrained_rotation"
    return "curobo" if controller == "traditional" else "local_policy"


def _infer_move_role(stage_dict: Mapping[str, Any], rels: Mapping[str, CompiledRelation]) -> str:
    if "approach" in rels:
        return "approach"
    if "held_transport" in rels:
        return "held_transport"
    motion_role = str(stage_dict.get("motion_role") or "")
    if motion_role in ("approach", "held_transport", "lift", "transport", "retreat"):
        return "retreat" if not rels else motion_role
    return "retreat"


def _topological_sort(stages: list[CompiledStage]) -> list[CompiledStage]:
    by_id = {s.stage_id: s for s in stages}
    in_deg = {s.stage_id: 0 for s in stages}
    adj: dict[str, list[str]] = {s.stage_id: [] for s in stages}
    for s in stages:
        for dep in s.depends_on:
            if dep not in by_id:
                raise ValueError(f"Unresolved dependency {dep!r} referenced by stage {s.stage_id!r}")
            adj[dep].append(s.stage_id)
            in_deg[s.stage_id] += 1

    ready = sorted(
        [sid for sid, deg in in_deg.items() if deg == 0],
        key=lambda sid: (by_id[sid].demo_frames[0], by_id[sid].demo_frames[1], sid),
    )
    ordered: list[CompiledStage] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(by_id[cur])
        for nxt in adj[cur]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda sid: (by_id[sid].demo_frames[0], by_id[sid].demo_frames[1], sid))

    if len(ordered) != len(stages):
        remaining = sorted(set(by_id) - {s.stage_id for s in ordered})
        raise ValueError(f"Cycle detected in compiled rule dependencies: {remaining}")
    return ordered


def compile_demo_rule(
    rule_data: Mapping[str, Any],
    *,
    source_path: str | None = None,
    allow_missing_geometry: bool = False,
) -> CompiledRulePlan:
    """Compile a parsed ``ula_demo_primitives_v1`` dict into ``CompiledRulePlan``."""
    schema = str(rule_data.get("schema", ""))
    if schema != SUPPORTED_SCHEMA:
        raise ValueError(f"Unsupported rule schema {schema!r}; expected {SUPPORTED_SCHEMA!r}")

    raw_stages = list(rule_data.get("stages") or [])
    if not raw_stages:
        raise ValueError("Rule contains no stages")

    bindings = dict(rule_data.get("bindings") or {})
    warnings: list[str] = []

    child_to_parent: dict[str, str] = {}
    top_ids = {str(s["id"]) for s in raw_stages}
    for s in raw_stages:
        sid = str(s["id"])
        for child in s.get("children") or []:
            child_to_parent[str(child["id"])] = sid

    def resolve_dep(dep_id: str, self_id: str) -> str | None:
        target = child_to_parent.get(dep_id, dep_id)
        if target == self_id:
            return None
        if target not in top_ids:
            raise ValueError(f"Stage {self_id!r} depends on unknown stage {dep_id!r}")
        return target

    stage_entries: list[dict[str, Any]] = []
    for s in raw_stages:
        sid = str(s["id"])
        primitive = str(s["primitive"])
        if primitive not in SUPPORTED_PRIMITIVES:
            raise ValueError(f"Stage {sid!r} uses unsupported P0 primitive {primitive!r}")

        missing = list(s.get("missing_geometry") or [])
        if missing and not allow_missing_geometry:
            raise ValueError(f"Stage {sid!r} has missing_geometry={missing}")

        frames_raw = s.get("demo_frames") or [0, 0]
        demo_frames = (int(frames_raw[0]), int(frames_raw[-1]))
        controller = str(s.get("controller") or "traditional")
        executor = str(s.get("executor") or _default_executor(primitive, controller))
        rels = _extract_relations(s)
        obj_goal = _extract_object_goal(s, bindings)

        deps: list[str] = []
        raw_deps = list(s.get("depends_on") or [])
        for child in s.get("children") or []:
            raw_deps.extend(child.get("depends_on") or [])
        for d in raw_deps:
            resolved = resolve_dep(str(d), sid)
            if resolved and resolved not in deps:
                deps.append(resolved)

        arm = s.get("arm")
        donor_arm = s.get("donor_arm")
        receiver_arm = s.get("receiver_arm")
        if primitive == "handover":
            role = "handover"
        elif primitive == "place":
            role = "place"
            if obj_goal is not None and "goal" not in rels:
                rels["goal"] = CompiledRelation(
                    name="goal",
                    reference=str(bindings.get("target") or "target"),
                    controlled=str(bindings.get("object") or "object"),
                    frame=demo_frames[1],
                    T_reference_controlled=obj_goal,
                    source="compiled_place_goal",
                )
        elif primitive == "move":
            role = _infer_move_role(s, rels)
        else:
            role = primitive

        gripper_open_val = float((s.get("geometry") or {}).get("gripper_open_value", 1.0))
        for child in s.get("children") or []:
            if child.get("primitive") == "release":
                gripper_open_val = float((child.get("geometry") or {}).get("gripper_open_value", gripper_open_val))

        delta_xyz = None
        motion = s.get("motion") or {}
        if motion.get("delta_world_xyz_m") is not None:
            v = [float(x) for x in motion["delta_world_xyz_m"]]
            if len(v) == 3:
                delta_xyz = (v[0], v[1], v[2])

        geom_stage = s.get("geometry") or {}
        rel_rot = None
        raw_rel_rot = geom_stage.get("relative_rotation")
        if isinstance(raw_rel_rot, list) and len(raw_rel_rot) == 4 and all(
            isinstance(row, list) and len(row) == 4 for row in raw_rel_rot
        ):
            rel_rot = [[float(x) for x in row] for row in raw_rel_rot]
        rel_rot_role = str(geom_stage.get("relative_rotation_role") or "") or None

        stage_entries.append(
            dict(
                stage_id=sid,
                primitive=primitive,
                controller=controller,
                executor=executor,
                role=role,
                arm=str(arm) if arm is not None else None,
                donor_arm=str(donor_arm) if donor_arm is not None else None,
                receiver_arm=str(receiver_arm) if receiver_arm is not None else None,
                object_name=str(s.get("object") or bindings.get("object") or "") or None,
                reference_name=str(s.get("reference") or bindings.get("target") or bindings.get("object") or "") or None,
                demo_frames=demo_frames,
                deps=deps,
                relations=rels,
                reusable_object_goal=obj_goal,
                gripper_open_value=gripper_open_val,
                motion_delta_xyz_m=delta_xyz,
                relative_rotation=rel_rot,
                relative_rotation_role=rel_rot_role,
                requires_reconcile=bool((s.get("geometry") or {}).get("requires_runtime_tool_frame_reconciliation", True)),
                child_ids=tuple(str(c["id"]) for c in (s.get("children") or [])),
            )
        )

    for e in stage_entries:
        if e["primitive"] == "handover":
            donor = e["donor_arm"]
            receiver = e["receiver_arm"]
            h_start = e["demo_frames"][0]
            donor_predecessors = [
                other for other in stage_entries
                if other["stage_id"] != e["stage_id"]
                and other["arm"] == donor
                and other["demo_frames"][0] <= h_start
                and other["primitive"] in ("move", "grasp")
            ]
            if donor_predecessors:
                donor_predecessors.sort(key=lambda x: (x["demo_frames"][1], x["demo_frames"][0]))
                latest_donor_entry = donor_predecessors[-1]
                latest_donor = latest_donor_entry["stage_id"]
                if latest_donor not in e["deps"]:
                    e["deps"].append(latest_donor)

                # Use the donor's final held-object goal for the handover
                # configuration. It is synchronous with the donor attachment
                # sampled at donor release start. A child move after receiver
                # closure may otherwise contribute a later object goal and
                # create a left/right pose combination never seen in the demo.
                donor_goal = latest_donor_entry.get("reusable_object_goal")
                if donor_goal is not None:
                    e["reusable_object_goal"] = donor_goal
                    if latest_donor_entry.get("reference_name") is not None:
                        e["reference_name"] = latest_donor_entry["reference_name"]
                    warnings.append(
                        f"Handover {e['stage_id']} uses synchronized object goal from {latest_donor}"
                    )
                for other in stage_entries:
                    if (
                        other["arm"] == receiver
                        and other["primitive"] == "move"
                        and other["role"] == "approach"
                        and other["demo_frames"][1] <= h_start
                    ):
                        if latest_donor not in other["deps"]:
                            other["deps"].append(latest_donor)
                            other_id = other["stage_id"]
                            warnings.append(f"Added handover barrier: {other_id} waits for {latest_donor}")
                        if other["stage_id"] not in e["deps"]:
                            e["deps"].append(other["stage_id"])

        elif e["primitive"] == "place":
            p_arm = e["arm"]
            p_start = e["demo_frames"][0]
            if not e["deps"]:
                candidates = [
                    other for other in stage_entries
                    if other["stage_id"] != e["stage_id"]
                    and other["demo_frames"][1] <= p_start
                    and (
                        (other["primitive"] == "handover" and other["receiver_arm"] == p_arm)
                        or (other["primitive"] == "grasp" and other["arm"] == p_arm)
                        or (other["primitive"] == "move" and other["arm"] == p_arm)
                    )
                ]
                if candidates:
                    candidates.sort(key=lambda x: (x["demo_frames"][1], x["demo_frames"][0]))
                    e["deps"].append(candidates[-1]["stage_id"])

    compiled_stages = [
        CompiledStage(
            stage_id=e["stage_id"],
            primitive=e["primitive"],
            controller=e["controller"],
            executor=e["executor"],
            role=e["role"],
            arm=e["arm"],
            donor_arm=e["donor_arm"],
            receiver_arm=e["receiver_arm"],
            object_name=e["object_name"],
            reference_name=e["reference_name"],
            demo_frames=e["demo_frames"],
            depends_on=tuple(e["deps"]),
            relations=e["relations"],
            reusable_object_goal=e["reusable_object_goal"],
            gripper_open_value=e["gripper_open_value"],
            motion_delta_xyz_m=e["motion_delta_xyz_m"],
            relative_rotation=e["relative_rotation"],
            relative_rotation_role=e["relative_rotation_role"],
            requires_runtime_tool_frame_reconciliation=e["requires_reconcile"],
            child_ids=e["child_ids"],
        )
        for e in stage_entries
    ]
    ordered = _topological_sort(compiled_stages)

    return CompiledRulePlan(
        schema=schema,
        task=str(rule_data.get("task", "unknown")),
        description=str(rule_data.get("description", "")),
        active_arms=tuple(str(a) for a in (rule_data.get("active_arms") or [])),
        bindings=bindings,
        stages=tuple(ordered),
        source_path=source_path,
        warnings=tuple(warnings),
    )


def load_and_compile_rule(
    rule_path: str | Path,
    *,
    allow_missing_geometry: bool = False,
) -> CompiledRulePlan:
    path = Path(rule_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return compile_demo_rule(data, source_path=str(path), allow_missing_geometry=allow_missing_geometry)


def remap_single_arm_plan(plan: CompiledRulePlan, target_arm: str) -> CompiledRulePlan:
    """Remap a single-arm frozen CompiledRulePlan to ``left`` or ``right`` symmetrically."""
    from dataclasses import replace

    if target_arm not in ("left", "right"):
        raise ValueError(f"Unsupported target_arm: {target_arm!r}")
    if len(plan.active_arms) != 1:
        return plan
    src_arm = plan.active_arms[0]
    if src_arm == target_arm:
        return plan

    src_ee = f"{src_arm}_recorded_ee"
    dst_ee = f"{target_arm}_recorded_ee"
    new_stages = []
    for stage in plan.stages:
        new_rels = {}
        for k, rel in stage.relations.items():
            new_rels[k] = replace(
                rel,
                reference=dst_ee if rel.reference == src_ee else rel.reference,
                controlled=dst_ee if rel.controlled == src_ee else rel.controlled,
            )
        new_stages.append(
            replace(
                stage,
                arm=target_arm if stage.arm == src_arm else stage.arm,
                donor_arm=target_arm if stage.donor_arm == src_arm else stage.donor_arm,
                receiver_arm=target_arm if stage.receiver_arm == src_arm else stage.receiver_arm,
                relations=new_rels,
            )
        )
    new_active = (target_arm,) if isinstance(plan.active_arms, tuple) else [target_arm]
    return replace(plan, active_arms=new_active, stages=tuple(new_stages) if isinstance(plan.stages, tuple) else new_stages)
