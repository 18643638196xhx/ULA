#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tra_pipe/RoboTwin')]
from diagnostics.handover_collision_attribution import attribute

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    attribute(args.audit_root,args.output)
