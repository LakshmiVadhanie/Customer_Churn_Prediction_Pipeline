"""
s3_uploader.py

Uploads local CSV log files to AWS S3 with multipart support,
progress tracking, and retry logic.
"""

import os
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class S3Uploader:
    def __init__(self, bucket: str, region: str = "us-east-1", max_workers: int = 8):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.max_workers = max_workers

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        try:
            file_size = os.path.getsize(local_path)
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=Path(local_path).name, leave=False) as pbar:
                self.s3.upload_file(
                    local_path,
                    self.bucket,
                    s3_key,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                )
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str) -> dict:
        local_path = Path(local_dir)
        files = list(local_path.glob("**/*.csv")) + list(local_path.glob("**/*.parquet"))

        logger.info(f"Uploading {len(files)} files to s3://{self.bucket}/{s3_prefix}")

        results = {"success": 0, "failed": 0}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}
            for f in files:
                rel_path = f.relative_to(local_path)
                s3_key = f"{s3_prefix.rstrip('/')}/{rel_path}"
                future = executor.submit(self.upload_file, str(f), s3_key)
                future_to_file[future] = f

            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                if future.result():
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    logger.warning(f"Upload failed: {filepath}")

        logger.info(f"Upload complete. Success: {results['success']}, Failed: {results['failed']}")
        return results

    def list_objects(self, prefix: str) -> list:
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def download_file(self, s3_key: str, local_path: str) -> bool:
        try:
            os.makedirs(Path(local_path).parent, exist_ok=True)
            self.s3.download_file(self.bucket, s3_key, local_path)
            return True
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Upload data files to AWS S3")
    parser.add_argument("--local", required=True, help="Local directory to upload")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="raw/customer_logs/", help="S3 key prefix")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--workers", type=int, default=8, help="Parallel upload workers")
    args = parser.parse_args()

    uploader = S3Uploader(bucket=args.bucket, region=args.region, max_workers=args.workers)
    uploader.upload_directory(args.local, args.prefix)


if __name__ == "__main__":
    main()
