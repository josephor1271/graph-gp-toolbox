#!/usr/bin/env python3
"""Fixed four-arm Stage 05 study; full measurements never run in CI."""
import argparse
from graphgp.benchmarking.stage05_coordinator import orchestrate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/chebyshev_stage05')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--profile-only',action='store_true')
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--scaled-candidate',action='store_true',help='only the gated, predeclared CSR candidate matrix')
    args=parser.parse_args()
    if sum((args.smoke,args.profile_only,args.scaled_candidate))>1:parser.error('choose one study mode')
    orchestrate(args.output,smoke=args.smoke,profile_only=args.profile_only,max_cases=args.max_cases,candidate=args.scaled_candidate)


if __name__=='__main__':main()
