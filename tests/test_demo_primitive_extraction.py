from pathlib import Path

import numpy as np
import pytest
import yaml

from adapters.demo_primitive_extraction import (
    ExtractionConfig, VisualTracks, detect_arm_events, extract_rule, pose_matrix,
)
from adapters.primitive_geometry import (
    apply_relative_pose, hand_goal_for_object_goal, relative_pose,
)
from scripts.extract_demo_primitives import infer_bindings


LIBRARY = yaml.safe_load((Path(__file__).resolve().parents[1]/'rules/primitive_library.yaml').read_text())


def track(count=180, x=0.):
    result = np.zeros((count, 7))
    result[:, :3] = [x, 0., 1.]
    result[:, 3] = 1.
    return result


def aperture(count, ramps):
    result = np.ones(count)
    for first, last, target in ramps:
        result[first:last+1] = np.linspace(result[first-1], target, last-first+1)
        result[last+1:] = target
    return result


def translate(poses, first, last, delta):
    base = poses[first, :3].copy()
    delta = np.asarray(delta)
    poses[first:last+1, :3] = base+np.linspace(0, 1, last-first+1)[:, None]*delta
    poses[last+1:, :3] += delta


def leaves(stages):
    for stage in stages:
        if stage.get('children'):
            yield from leaves(stage['children'])
        else:
            yield stage


def compile_demo(poses, grippers, task='unknown', description='', **kwargs):
    return extract_rule(task, description, poses, grippers, 15., LIBRARY, **kwargs)


def pick_place(count=180):
    poses = track(count)
    translate(poses, 5, 25, [.1, 0, 0])
    translate(poses, 32, 40, [0, 0, -.03])
    translate(poses, 65, 75, [0, 0, .08])
    translate(poses, 85, 105, [.2, 0, 0])
    translate(poses, 115, 125, [0, 0, -.08])
    return poses, aperture(count, [(45, 55, 0.), (135, 145, 1.)])


def test_repeated_grasps_and_releases_are_not_limited_to_one():
    count = 240
    poses = track(count)
    for start in (5, 65, 125):
        translate(poses, start, start+10, [.05, 0, 0])
        translate(poses, start+30, start+40, [0, .05, 0])
    gripper = aperture(count, [(20, 25, 0), (50, 55, 1),
                               (80, 85, 0), (110, 115, 1),
                               (140, 145, 0), (170, 175, 1)])
    rule = compile_demo({'left': poses}, {'left': gripper}, 'place_objects', 'place objects on a support')
    values = list(leaves(rule['stages']))
    assert sum(stage['primitive'] == 'grasp' for stage in values) == 3
    assert sum(stage['primitive'] == 'release' for stage in values) == 3
    assert all(stage['demo_frames'][0] <= stage['demo_frames'][1] for stage in values)


def test_place_is_one_composite_whose_children_are_not_executed_twice():
    poses, gripper = pick_place()
    rule = compile_demo({'left': poses}, {'left': gripper}, 'place_object', 'place the bottle on the pad',
                        object_name='bottle', target='pad')
    groups = [stage for stage in rule['stages'] if stage['primitive'] == 'place']
    assert len(groups) == 1
    child_ids = {stage['id'] for stage in groups[0]['children']}
    root_ids = {stage['id'] for stage in rule['stages']}
    assert not child_ids & root_ids
    assert groups[0]['children'][-1]['primitive'] == 'release'
    assert groups[0]['object_goal']['T_reference_object'] is None
    assert rule['runtime_ready'] is False


def test_missing_release_does_not_invent_place_or_open_gripper():
    poses, _ = pick_place()
    gripper = aperture(len(poses), [(45, 55, 0)])
    rule = compile_demo({'left': poses}, {'left': gripper}, 'place_object', 'place object')
    assert not any(stage['primitive'] in ('place', 'release') for stage in rule['stages'])


def dual_demo(handover):
    count = 200
    left, right = track(count, -.2), track(count, .2)
    translate(left, 5, 20, [.1, 0, 0])
    translate(left, 50, 75, [.1, 0, .1])
    translate(right, 55, 85, [-.2, 0, .1])
    translate(right, 145, 165, [.15, .1, -.05])
    left_gripper = aperture(count, [(30, 40, 0), (120, 130, 1)])
    right_gripper = aperture(count, [(95, 105, 0), (175, 185, 1)])
    return compile_demo({'left': left, 'right': right}, {'left': left_gripper, 'right': right_gripper},
                        'handover_object' if handover else 'pick_two_objects',
                        'handover the same block' if handover else 'pick two independent bottles')


def test_handover_adds_receive_before_donor_release_dependency():
    rule = dual_demo(True)
    group = next(stage for stage in rule['stages'] if stage['primitive'] == 'handover')
    receiver, donor_release = group['children']
    assert receiver['arm'] == 'right' and donor_release['arm'] == 'left'
    assert receiver['demo_frames'][1] <= donor_release['demo_frames'][0]
    assert [receiver['id'], donor_release['id']] in rule['execution_graph']['edges']
    assert group['same_object_visually_verified'] is False


def test_independent_dual_arm_grasps_are_not_mislabeled_as_handover():
    rule = dual_demo(False)
    assert set(rule['active_arms']) == {'left', 'right'}
    assert not any(stage['primitive'] == 'handover' for stage in rule['stages'])
    assert not any(left.startswith('left_') and right.startswith('right_')
                   or left.startswith('right_') and right.startswith('left_')
                   for left, right in rule['execution_graph']['edges'])


def test_static_visual_pose_is_reused_only_when_declared_static():
    pose = np.eye(4)
    assert VisualTracks(poses={'pad': {0: pose}}).pose('pad', 12) is None
    assert np.allclose(VisualTracks(poses={'pad': {0: pose}}, static_entities=['pad']).pose('pad', 12), pose)
    assert VisualTracks(initial_anchors={'bottle': pose}).pose('bottle', 12) is None


def test_relative_keyframes_reconstruct_recorded_ee_pose():
    poses, gripper = pick_place()
    anchor = np.eye(4)
    anchor[:3, 3] = [.1, .2, .8]
    visual = VisualTracks(initial_anchors={'bottle': anchor})
    rule = compile_demo({'left': poses}, {'left': gripper}, object_name='bottle', visual=visual)
    for keyframe in rule['keyframes'].values():
        relative = keyframe['relative_to_initial_visual_anchors']['bottle']
        assert np.allclose(anchor@relative, keyframe['T_world_recorded_ee'])


def test_place_goal_comes_from_object_pose_not_hand_pose():
    poses, gripper = pick_place()
    bottle, pad = np.eye(4), np.eye(4)
    bottle[:3, 3], pad[:3, 3] = [.5, .4, .8], [.49, .4, .79]
    visual = VisualTracks(poses={'bottle': {135: bottle}, 'pad': {0: pad}}, static_entities=['pad'])
    rule = compile_demo({'left': poses}, {'left': gripper}, 'place_bottle', 'place bottle on pad',
                        object_name='bottle', target='pad', visual=visual)
    group = next(stage for stage in rule['stages'] if stage['primitive'] == 'place')
    relative = np.asarray(group['object_goal']['T_reference_object'])
    assert np.allclose(relative, np.linalg.inv(pad)@bottle)
    assert not np.allclose(relative, np.linalg.inv(pad)@pose_matrix(poses[135]))


def test_contact_label_without_visual_evidence_remains_candidate():
    poses, gripper = pick_place()
    rule = compile_demo({'left': poses}, {'left': gripper}, 'beat_target', 'strike target with tool')
    strike = next(stage for stage in leaves(rule['stages']) if stage['primitive'] == 'strike')
    assert strike['evidence_level'] == 'task_conditioned_candidate'
    assert strike['physical_contact_confirmed'] is False
    assert 'verified_contact_reference_and_tool_contact_frame' in strike['missing_geometry']


def test_pure_rotation_is_detected_and_quaternion_sign_flip_is_not_motion():
    poses = track(80)
    theta = np.r_[np.zeros(20), np.linspace(0, np.pi/2, 20), np.full(40, np.pi/2)]
    poses[:, 3], poses[:, 6] = np.cos(theta/2), np.sin(theta/2)
    rule = compile_demo({'left': poses}, {'left': np.ones(80)}, 'rotate_target', 'rotate target')
    assert any(stage['primitive'] == 'rotate' for stage in leaves(rule['stages']))
    poses = track(80)
    poses[::2, 3:] *= -1
    detected = detect_arm_events(poses, np.ones(80), 15.)
    assert detected['bouts'] == []


def test_open_without_gripper_events_still_has_open_composite():
    poses = track(120)
    translate(poses, 5, 30, [.1, 0, 0])
    translate(poses, 55, 85, [0, -.1, .05])
    rule = compile_demo({'left': poses}, {'left': np.ones(120)}, 'open_lid', 'open the lid')
    group = next(stage for stage in rule['stages'] if stage['primitive'] == 'open')
    assert all(stage['primitive'] == 'move' for stage in group['children'])
    assert 'articulation_axis_or_path' in group['missing_geometry']


def test_gripper_calibration_can_be_reversed():
    poses = track(80)
    gripper = aperture(80, [(20, 30, 0), (50, 60, 1)])
    result = detect_arm_events(poses, 1-gripper, 15.,
                               ExtractionConfig(gripper_open_value=0., gripper_closed_value=1.))
    assert [event['kind'] for event in result['events']] == ['gripper_close', 'gripper_open']


def test_invalid_unsynchronized_pose_tracks_are_rejected():
    with pytest.raises(ValueError, match='synchronized'):
        compile_demo({'left': track(80), 'right': track(90)},
                      {'left': np.ones(80), 'right': np.ones(90)})


def test_visual_comotion_is_recorded_without_inventing_a_gripper_closure():
    poses = track(80)
    translate(poses, 20, 40, [.1, 0, 0])
    objects = {}
    for frame in range(80):
        obj = pose_matrix(poses[frame])
        obj[:3, 3] += [0, 0, -.05]
        objects[frame] = obj
    visual = VisualTracks(poses={'bottle': objects})
    rule = compile_demo({'left': poses}, {'left': np.ones(80)}, object_name='bottle', visual=visual)
    assert any(event['kind'] == 'object_follows_hand' for event in rule['events'])
    assert not any(stage['primitive'] == 'grasp' for stage in leaves(rule['stages']))


def test_contact_proximity_uses_calibrated_tool_frame_and_measured_depth():
    poses = track(5)
    surface = np.array([[0., 0., .8], [.02, 0., .8], [0., .02, .8]])
    missing = VisualTracks(points={'button': {0: surface}})
    assert not missing.contact_proximity('button', poses, 0, 0)['available']
    transform = np.eye(4)
    transform[2, 3] = -.2
    visual = VisualTracks(points={'button': {0: surface}}, metadata={'T_recorded_ee_contact': transform})
    evidence = visual.contact_proximity('button', poses, 0, 0)
    assert evidence['supported']
    assert evidence['physical_contact_proven'] is False


def test_multiple_objects_are_associated_per_arm_instead_of_reusing_one_binding():
    count = 180
    left, lg = pick_place(count)
    right, rg = pick_place(count)
    left[:, 0] -= .3
    right[:, 0] += .3
    objects = {}
    for name, poses in [('first', left), ('second', right)]:
        objects[name] = {}
        for frame in range(count):
            matrix = pose_matrix(poses[frame])
            matrix[2, 3] -= .05
            objects[name][frame] = matrix
    rule = compile_demo({'left': left, 'right': right}, {'left': lg, 'right': rg},
                        'pick_two_bottles', 'pick two separate bottles', visual=VisualTracks(poses=objects))
    grasps = {stage['arm']: stage['object'] for stage in leaves(rule['stages']) if stage['primitive'] == 'grasp'}
    assert grasps == {'left': 'first', 'right': 'second'}


def test_preparatory_close_for_direct_press_is_not_a_vla_grasp():
    poses = track(120)
    translate(poses, 5, 25, [.1, 0, 0])
    translate(poses, 60, 80, [0, 0, -.02])
    gripper = aperture(120, [(35, 45, 0)])
    rule = compile_demo({'left': poses}, {'left': gripper}, 'press_button', 'press the button')
    values = list(leaves(rule['stages']))
    assert not any(stage['primitive'] == 'grasp' for stage in values)
    press = next(stage for stage in values if stage['primitive'] == 'press')
    assert press['gripper_configuration'][0]['demo_frames'] == [35, 45]
    assert press['controller'] == 'traditional'


def test_mixed_open_and_place_goals_do_not_swallow_object_arm_into_open():
    left, lg = pick_place(180)
    right = track(180, .3)
    translate(right, 10, 30, [0, .1, 0])
    translate(right, 80, 100, [0, -.1, 0])
    rg = aperture(180, [(45, 55, 0)])
    rule = compile_demo({'left': left, 'right': right}, {'left': lg, 'right': rg},
                        'put_item_drawer', 'open the drawer with one hand and place the item inside')
    opening = [stage for stage in rule['stages'] if stage['primitive'] == 'open']
    placement = [stage for stage in rule['stages'] if stage['primitive'] == 'place']
    assert len(opening) == len(placement) == 1
    assert opening[0]['arm'] == 'right' and placement[0]['arm'] == 'left'


def test_receiver_carry_crossing_transfer_boundary_is_split_before_place():
    count = 200
    left, right = track(count, -.2), track(count, .2)
    translate(left, 5, 20, [.1, 0, 0])
    translate(left, 50, 75, [.1, 0, .1])
    translate(right, 55, 85, [-.2, 0, .1])
    translate(right, 106, 165, [.15, .1, -.05])
    lg = aperture(count, [(30, 40, 0), (120, 130, 1)])
    rg = aperture(count, [(95, 105, 0), (175, 185, 1)])
    rule = compile_demo({'left': left, 'right': right}, {'left': lg, 'right': rg},
                        'handover_object', 'handover the block and place on support')
    transfer = next(stage for stage in rule['stages'] if stage['primitive'] == 'handover')
    placement = next(stage for stage in rule['stages'] if stage['primitive'] == 'place')
    assert placement['demo_frames'][0] >= transfer['demo_frames'][1]
    assert any(stage['primitive'] == 'move' for stage in transfer['children'])
    donor_release = next(stage for stage in transfer['children'] if stage['primitive'] == 'release')
    assert [donor_release['id'], placement['children'][0]['id']] in rule['execution_graph']['edges']


def test_generic_relative_pose_moves_with_current_object():
    demo_object, demo_hand = np.eye(4), np.eye(4)
    demo_object[:3, 3] = [.4, -.2, .8]
    demo_hand[:3, 3] = [.4, -.3, .9]
    relation = relative_pose(demo_object, demo_hand)
    current_object = demo_object.copy()
    current_object[:3, 3] += [.25, .1, 0.]
    current_hand = apply_relative_pose(current_object, relation)
    assert np.allclose(current_hand[:3, 3], demo_hand[:3, 3] + [.25, .1, 0.])


def test_generic_place_goal_compensates_current_grasp_attachment():
    target, target_object, ee_object = np.eye(4), np.eye(4), np.eye(4)
    target[:3, 3] = [.5, .2, .75]
    target_object[:3, 3] = [0., 0., .04]
    ee_object[:3, 3] = [0., 0., -.10]
    hand = hand_goal_for_object_goal(target, target_object, ee_object)
    achieved_object = hand @ ee_object
    assert np.allclose(achieved_object, target @ target_object)


def test_grasp_rule_records_object_relative_precontact_and_attachment():
    poses, gripper = pick_place()
    object_poses = {}
    for frame in range(len(poses)):
        value = pose_matrix(poses[frame])
        value[:3, 3] += [0., 0., -.06]
        object_poses[frame] = value
    visual = VisualTracks(poses={'bottle': object_poses})
    rule = compile_demo({'left': poses}, {'left': gripper}, object_name='bottle', visual=visual)
    grasp = next(stage for stage in leaves(rule['stages']) if stage['primitive'] == 'grasp')
    names = {relation['name'] for relation in grasp['geometry']['demonstrated_relations']}
    assert names == {'precontact', 'attachment'}
    attachment = next(value for value in grasp['geometry']['demonstrated_relations']
                      if value['name'] == 'attachment')
    frame = attachment['frame']
    assert np.allclose(pose_matrix(poses[frame]) @ np.asarray(attachment['T_reference_controlled']),
                       object_poses[frame])


def test_handover_uses_the_same_generic_object_relative_geometry_contract():
    count = 200
    left, right = track(count, -.2), track(count, .2)
    translate(left, 5, 20, [.1, 0, 0])
    translate(left, 50, 75, [.1, 0, .1])
    translate(right, 55, 85, [-.2, 0, .1])
    translate(right, 145, 165, [.15, .1, -.05])
    lg = aperture(count, [(30, 40, 0), (120, 130, 1)])
    rg = aperture(count, [(95, 105, 0), (175, 185, 1)])
    block = {}
    for frame in range(count):
        arm = left if frame <= 120 else right
        value = pose_matrix(arm[frame])
        value[:3, 3] += [0., 0., -.05]
        block[frame] = value
    rule = compile_demo(
        {'left': left, 'right': right}, {'left': lg, 'right': rg},
        'handover_block', 'handover the block and place it on the pad',
        object_name='block', target='pad', visual=VisualTracks(
            poses={'block': block}, metadata={'identity_continuity': {
                'block': {'method': 'same_measured_template_track'}}}))
    transfer = next(stage for stage in rule['stages'] if stage['primitive'] == 'handover')
    names = {item['name'] for item in transfer['geometry']['demonstrated_relations']}
    assert names == {'receiver_precontact', 'receiver_attachment', 'donor_attachment'}
    assert transfer['same_object_visually_verified'] is True
    assert transfer['geometry']['runtime_rule'] == 'receiver_close_must_finish_before_donor_release'
    right_approach = next(stage for stage in leaves(rule['stages'])
                          if stage['arm'] == 'right' and stage['primitive'] == 'move'
                          and stage['demo_frames'][1] <= 95)
    assert right_approach['geometry']['reference_scope'] == 'current_visual_pose'
    postrelease = [stage for stage in leaves(rule['stages'])
                   if stage['primitive'] == 'move' and stage.get('gripper_state') == 'open'
                   and stage['demo_frames'][0] > 130]
    assert all(not stage['geometry']['demonstrated_relations'] for stage in postrelease)


def test_role_name_binding_is_generic_for_object_and_target_aliases():
    assert infer_bindings(
        'use the left arm to grasp the red block, handover it and place it on the blue pad',
        ['box', 'target_box'], None, None) == ('box', 'target_box')
