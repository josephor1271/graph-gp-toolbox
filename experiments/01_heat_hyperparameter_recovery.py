"""Reproducible dense-exact / dense-polynomial / sparse-polynomial comparison."""
import argparse
from graphgp.recovery import run_study, plot_results, summarize

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/heat_recovery')
    parser.add_argument('--steps',type=int,default=180)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--learn-amplitude',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.plot_only:
        summarize(args.output)
        plot_results(args.output)
    else:
        run_study(args.output,smoke=args.smoke,steps=5 if args.smoke else args.steps,
                  learn_amplitude=args.learn_amplitude)
