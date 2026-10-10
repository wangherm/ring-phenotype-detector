import argparse
from pathlib import Path
from .io import audit
from .pipeline import load_config,run_segment,run_propose
from .annotations import review_export,validate_annotations

def main():
    parser=argparse.ArgumentParser(description='P0/P1 color-independent ring candidate baseline; no classifier training')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit');p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    for name in ['segment','propose','predict']:
        p=sub.add_parser(name);p.add_argument('--manifest',required=True);p.add_argument('--config',default='configs/default.yaml');p.add_argument('--output',required=True)
        if name!='segment':p.add_argument('--segments',required=True)
        if name=='predict':p.add_argument('--baseline-only',action='store_true',required=True)
    p=sub.add_parser('review-export');p.add_argument('--proposals',required=True);p.add_argument('--output',required=True);p.add_argument('--per-fov',type=int,default=6)
    p=sub.add_parser('review-check');p.add_argument('--annotations',required=True);p.add_argument('--cells',required=True)
    a=parser.parse_args()
    if a.command=='audit':audit(a.input,a.output)
    elif a.command=='segment':run_segment(a.manifest,load_config(a.config),a.output)
    elif a.command in ('propose','predict'):run_propose(a.manifest,a.segments,load_config(a.config),a.output)
    elif a.command=='review-export':review_export(a.proposals,a.output,a.per_fov)
    elif a.command=='review-check':validate_annotations(a.annotations,a.cells)

if __name__=='__main__':main()
