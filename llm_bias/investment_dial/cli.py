"""Independent phase commands; no implicit experiment chaining."""
import argparse

from .pipeline import run_check, run_screen, run_calibration, run_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Investment-bias dial local V1 method reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("run-check", help="one-prompt engineering check, not a research result")
    check.add_argument("--input", required=True)
    check.add_argument("--max-new-tokens", type=int, default=128)
    check.add_argument("--cpu-bf16", action="store_true")
    screen = sub.add_parser("run-screen", help="screen gradients using only screen companies")
    screen.add_argument("--input", required=True)
    screen.add_argument("--top-k", type=int, default=3)
    screen.add_argument("--repeats", type=int, default=2)
    screen.add_argument("--seed", type=int, default=42)
    screen.add_argument("--layers", type=int, nargs="+")
    calibration = sub.add_parser("run-calibration", help="fit A curves, rank candidates on B")
    calibration.add_argument("--source-run", required=True)
    calibration.add_argument("--deltas", type=float, nargs="+", default=[-8., -4., 0., 4., 8.])
    calibration.add_argument("--targets", type=float, nargs="+", default=[-.3, 0., .3])
    calibration.add_argument("--minimum-rate", type=float, default=.9)
    calibration.add_argument("--max-new-tokens", type=int, default=256)
    evaluation = sub.add_parser("run-evaluation", help="test frozen selection on untouched companies")
    evaluation.add_argument("--source-run", required=True)
    for command in (check, screen, calibration, evaluation):
        command.add_argument("--model", default=".cache/models/qwen3.5-4b")
        command.add_argument("--run-id", required=True)
        command.add_argument("--artifact-root", default="artifacts")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    args["model_path"] = args.pop("model")
    if command in {"run-check", "run-screen"}:
        args["input_path"] = args.pop("input")
        path = run_check(**args) if command == "run-check" else run_screen(**args)
    elif command == "run-calibration":
        path = run_calibration(**args)
    else:
        path = run_evaluation(**args)
    print(path)
