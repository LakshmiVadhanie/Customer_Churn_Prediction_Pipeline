"""
pipeline.py

End-to-end pipeline orchestrator for the customer churn prediction system.
Coordinates: data generation -> S3 upload -> MapReduce -> Hive -> PySpark training.
"""

import os
import sys
import argparse
import logging
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("pipeline")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_step(name: str, cmd: list, env: dict = None) -> bool:
    logger.info(f"[STEP] {name}")
    result = subprocess.run(cmd, env={**os.environ, **(env or {})})
    if result.returncode != 0:
        logger.error(f"[FAILED] {name} exited with code {result.returncode}")
        return False
    logger.info(f"[OK] {name}")
    return True


def run_pipeline(config: dict, env: str, records: int):
    start = datetime.now()
    logger.info(f"Starting Customer Churn Prediction Pipeline [{env}]")
    logger.info("=" * 60)

    data_cfg = config.get("data", {})
    aws_cfg = config.get("aws", {})
    spark_cfg = config.get("spark", {})
    model_cfg = config.get("model", {})

    local_data = data_cfg.get("local_data_path", "data/sample/")
    local_features = "data/processed/features/"
    model_output = spark_cfg.get("model_path", "models/gbt_churn_model")

    Path(local_data).mkdir(parents=True, exist_ok=True)
    Path(local_features).mkdir(parents=True, exist_ok=True)
    Path(model_output).mkdir(parents=True, exist_ok=True)

    steps = []

    # Step 1: Generate sample data
    steps.append((
        "Generate Customer Log Data",
        [
            sys.executable, "src/aws/s3_data_generator.py",
            "--records", str(records),
            "--output", local_data,
        ]
    ))

    if env == "production":
        # Step 2a: Upload to S3
        steps.append((
            "Upload Logs to S3",
            [
                sys.executable, "src/aws/s3_uploader.py",
                "--local", local_data,
                "--bucket", aws_cfg.get("s3_bucket", "your-bucket"),
                "--prefix", aws_cfg.get("s3_logs_prefix", "raw/customer_logs/"),
            ]
        ))

        # Step 3a: Run Hadoop MapReduce
        steps.append((
            "MapReduce Feature Aggregation (Hadoop)",
            [
                sys.executable, "src/mapreduce/runner.py",
                "--mode", "hadoop",
                "--input", f"s3://{aws_cfg['s3_bucket']}/{aws_cfg.get('s3_logs_prefix', 'raw/')}",
                "--output", f"s3://{aws_cfg['s3_bucket']}/{aws_cfg.get('s3_features_prefix', 'processed/')}",
            ]
        ))
    else:
        # Step 2b: Local MapReduce simulation
        steps.append((
            "MapReduce Feature Aggregation (Local)",
            [
                sys.executable, "src/mapreduce/runner.py",
                "--mode", "local",
                "--input", local_data,
                "--output", local_features,
            ]
        ))

    # Step 4: Train PySpark GBT model
    steps.append((
        "Train GBT Model (PySpark)",
        [
            sys.executable, "src/spark/train_model.py",
            "--input", local_features if env == "local" else f"s3://{aws_cfg['s3_bucket']}/{aws_cfg.get('s3_features_prefix', 'processed/')}",
            "--model-output", model_output,
            "--master", spark_cfg.get("master", "local[*]"),
            "--num-trees", str(model_cfg.get("num_trees", 150)),
            "--max-depth", str(model_cfg.get("max_depth", 6)),
            "--learning-rate", str(model_cfg.get("learning_rate", 0.1)),
            "--subsample-rate", str(model_cfg.get("subsample_rate", 0.8)),
        ]
    ))

    # Run all steps
    failed = []
    for name, cmd in steps:
        success = run_step(name, cmd)
        if not success:
            failed.append(name)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Pipeline finished in {elapsed:.1f}s")
    logger.info(f"Steps completed: {len(steps) - len(failed)}/{len(steps)}")
    if failed:
        logger.error(f"Failed steps: {failed}")
        sys.exit(1)
    else:
        logger.info("All steps completed successfully.")
        logger.info(f"Model saved to: {model_output}")
        logger.info("Launch dashboard: python dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--env", choices=["local", "production"], default="local")
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    parser.add_argument("--records", type=int, default=100000, help="Records to generate (local mode)")
    parser.add_argument("--s3-bucket", help="Override S3 bucket name")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        example = Path("config/config.example.yaml")
        if example.exists():
            import shutil
            shutil.copy(example, config_path)
            logger.info(f"Created {config_path} from example. Edit it to configure your environment.")

    config = load_config(str(config_path))

    if args.s3_bucket:
        config.setdefault("aws", {})["s3_bucket"] = args.s3_bucket

    run_pipeline(config, args.env, args.records)


if __name__ == "__main__":
    main()
