"""Validate paired training data. Does not delete or skip problematic data."""
import argparse, json
from ml.config import DEFAULT_CONFIG
from ml.dataset import validate_dataset
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-dir",type=str); args=parser.parse_args(); config=DEFAULT_CONFIG
    if args.dataset_dir: config.dataset_dir=__import__('pathlib').Path(args.dataset_dir)
    try:
        _,report=validate_dataset(config); print(json.dumps(report,indent=2)); print("DATASET VALIDATION PASSED")
    except ValueError as exc: print(f"DATASET VALIDATION FAILED\n{exc}"); raise SystemExit(1)
if __name__=="__main__": main()
