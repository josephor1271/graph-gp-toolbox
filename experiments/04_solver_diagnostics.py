"""Run only Stage 04's predeclared profile/probe-seed diagnostic sweep."""
import argparse
from graphgp.benchmarking.coordinator import orchestrate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/solver_stage04")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    orchestrate(args.output, smoke=args.smoke, max_cases=args.max_cases, solver_only=True)
