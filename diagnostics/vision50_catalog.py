"""Explicit evaluation targets, not branches in the localization algorithm.

Selectors are used by the scene harness to supply the DECLARED known-instance
CAD and, only after prediction, to collect independent renderer labels.
They never supply a pose, pixel, ROI, contact or functional-point annotation.
"""
import re


TASK_TARGETS={
    'adjust_bottle':['bottle'],
    'beat_block_hammer':['hammer','block'],
    'blocks_ranking_rgb':['block1','block2','block3'],
    'blocks_ranking_size':['block1','block2','block3'],
    'click_alarmclock':['alarm'],
    'click_bell':['bell'],
    'dump_bin_bigbin':['deskbin','dustbin'],
    'grab_roller':['roller'],
    'handover_block':['box','target_box'],
    'handover_mic':['microphone'],
    'hanging_mug':['mug','rack'],
    'lift_pot':['pot'],
    'move_can_pot':['can','pot'],
    'move_pillbottle_pad':['pillbottle','pad'],
    'move_playingcard_away':['playingcards'],
    'move_stapler_pad':['stapler','pad'],
    'open_laptop':['laptop'],
    'open_microwave':['microwave'],
    'pick_diverse_bottles':['bottle1','bottle2'],
    'pick_dual_bottles':['bottle1','bottle2'],
    'place_a2b_left':['object','target_object'],
    'place_a2b_right':['object','target_object'],
    'place_bread_basket':['bread.*','breadbasket'],
    'place_bread_skillet':['bread','skillet'],
    'place_burger_fries':['hamburg','frenchfries','tray'],
    'place_can_basket':['can','basket'],
    'place_cans_plasticbox':['object1','object2','plasticbox'],
    'place_container_plate':['container','plate'],
    'place_dual_shoes':['left_shoe','right_shoe','shoe_box'],
    'place_empty_cup':['cup','coaster'],
    'place_fan':['fan','pad'],
    'place_mouse_pad':['mouse','target'],
    'place_object_basket':['object','basket'],
    'place_object_scale':['object','scale'],
    'place_object_stand':['object','displaystand'],
    'place_phone_stand':['phone','stand'],
    'place_shoe':['shoe','target_block'],
    'press_stapler':['stapler'],
    'put_bottles_dustbin':['bottles.*','dustbin'],
    'put_object_cabinet':['object','cabinet'],
    'rotate_qrcode':['qrcode'],
    'scan_object':['scanner','object'],
    'shake_bottle':['bottle'],
    'shake_bottle_horizontally':['bottle'],
    'stack_blocks_three':['block1','block2','block3'],
    'stack_blocks_two':['block1','block2'],
    'stack_bowls_three':['bowl1','bowl2','bowl3'],
    'stack_bowls_two':['bowl1','bowl2'],
    'stamp_seal':['seal','target'],
    'turn_switch':['switch'],
}

CAPTIONS={
    'alarm-clock':('alarm clock',), 'breadbasket':('bread basket', 'basket'),
    'displaystand':('display stand',), 'electronicscale':('kitchen scale', 'digital scale'),
    'french-fries':('french fries',), 'hamburg':('hamburger',),
    'kitchenpot':('cooking pot', 'pot'), 'paymentsign':('QR code sign',),
    'phonestand':('phone stand',), 'pillbottle':('pill bottle',),
    'plasticbox':('plastic box',), 'playingcards':('playing cards', 'deck of cards'),
    'rack':('mug rack', 'rack'), 'sauce-can':('can',), 'shoe-box':('shoe box',),
    'tabletrashbin':('small trash bin', 'trash bin'), 'toycar':('toy car',),
    'woodenblock':('wooden block',), 'tea-box':('tea box',),
    'coffee-box':('coffee box',), 'dustbin':('trash bin',), 'coaster':('coaster',),
}


def targets(task,name):
    """Explicit public object roles. No source-code eval or positional sorting."""
    if name not in TASK_TARGETS:raise ValueError('Task absent from the frozen 50-task catalog')
    result=[]
    for selector in TASK_TARGETS[name]:
        if selector.endswith('.*'):
            values=getattr(task,selector[:-2])
            if not isinstance(values,(list,tuple)) or not values:raise ValueError('Missing task object collection: '+selector)
            result.extend((selector[:-2]+'_'+str(i),value) for i,value in enumerate(values))
        else:
            if not re.fullmatch('[a-z][a-z0-9_]*',selector):raise ValueError('Unsafe target selector')
            result.append((selector,getattr(task,selector)))
    return result


def captions(modelname):
    name=re.sub(r'^\d+_','',modelname)
    return CAPTIONS.get(name,(name.replace('_',' ').replace('-',' '),))
