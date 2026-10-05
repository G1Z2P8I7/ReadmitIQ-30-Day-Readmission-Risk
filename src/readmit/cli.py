"""CLI entrypoint for ReadmitIQ."""

import argparse
import logging
import subprocess
import sys

from readmit import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cli")

def cmd_data(args):
    from readmit.data import run_data_step
    return run_data_step(args)

def cmd_features(args):
    from readmit.features import run_features_step
    return run_features_step(args)

def cmd_train(args):
    from readmit.models import run_train_step
    return run_train_step(args)

def cmd_evaluate(args):
    from readmit.evaluation import run_evaluate_step
    return run_evaluate_step(args)

def cmd_explain(args):
    from readmit.explain import run_explain_step
    return run_explain_step(args)

def cmd_fairness(args):
    from readmit.fairness import run_fairness_step
    return run_fairness_step(args)

def cmd_final(args):
    from readmit.final_eval import run_final_step
    return run_final_step(args)

def cmd_app(args):
    cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
    return subprocess.run(cmd, check=False).returncode

def cmd_api(args):
    import uvicorn
    return uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(description="ReadmitIQ CLI")
    parser.add_argument("--version", action="version", version=f"ReadmitIQ {__version__}")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    parser_data = subparsers.add_parser("data", help="Load, audit, and prepare cohort/splits")
    parser_data.set_defaults(func=cmd_data)

    parser_feat = subparsers.add_parser("features", help="Extract and prepare features")
    parser_feat.set_defaults(func=cmd_features)

    parser_train = subparsers.add_parser("train", help="Train baselines, LR, XGBoost, imbalance ablation")
    parser_train.set_defaults(func=cmd_train)

    parser_eval = subparsers.add_parser("evaluate", help="Evaluate models, calibration, and capacity")
    parser_eval.set_defaults(func=cmd_evaluate)

    parser_expl = subparsers.add_parser("explain", help="SHAP and interpretability")
    parser_expl.set_defaults(func=cmd_explain)

    parser_fair = subparsers.add_parser("fairness", help="Fairness audit and mitigation")
    parser_fair.set_defaults(func=cmd_fairness)

    parser_final = subparsers.add_parser("final", help="Score test set once and lock metrics")
    parser_final.add_argument("--force", action="store_true", help="Force re-scoring of test set")
    parser_final.set_defaults(func=cmd_final)

    parser_app = subparsers.add_parser("app", help="Run Streamlit dashboard")
    parser_app.set_defaults(func=cmd_app)

    parser_api = subparsers.add_parser("api", help="Run FastAPI service")
    parser_api.set_defaults(func=cmd_api)

    args = parser.parse_args()
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main() or 0)
