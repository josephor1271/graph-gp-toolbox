"""Run the fixed Stage 04 isolated-operation benchmark."""
import argparse
from graphgp.benchmarking.coordinator import orchestrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/scaling_stage04")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    orchestrate(args.output, smoke=args.smoke, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
