"""
runner.py

Orchestrates the Hadoop Streaming MapReduce job for customer churn
feature aggregation. Supports both local simulation and real Hadoop cluster.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAPPER_PATH = Path(__file__).parent / "mapper.py"
REDUCER_PATH = Path(__file__).parent / "reducer.py"


class HadoopStreamingRunner:
    def __init__(self, hadoop_home: str = None, streaming_jar: str = None):
        self.hadoop_home = hadoop_home or os.environ.get("HADOOP_HOME", "/opt/hadoop")
        self.streaming_jar = streaming_jar or self._find_streaming_jar()

    def _find_streaming_jar(self) -> str:
        jar_paths = [
            f"{self.hadoop_home}/share/hadoop/tools/lib/hadoop-streaming-*.jar",
        ]
        import glob
        for pattern in jar_paths:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        return "hadoop-streaming.jar"

    def run(self, input_path: str, output_path: str, num_reducers: int = 4) -> int:
        cmd = [
            "hadoop", "jar", self.streaming_jar,
            "-D", f"mapreduce.job.reduces={num_reducers}",
            "-D", "mapreduce.job.name=CustomerChurnFeatureEngineering",
            "-D", "mapreduce.map.memory.mb=2048",
            "-D", "mapreduce.reduce.memory.mb=4096",
            "-input", input_path,
            "-output", output_path,
            "-mapper", f"python3 {MAPPER_PATH}",
            "-reducer", f"python3 {REDUCER_PATH}",
            "-file", str(MAPPER_PATH),
            "-file", str(REDUCER_PATH),
        ]

        logger.info("Submitting MapReduce job...")
        logger.info(f"Input:  {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False)
        return result.returncode


class LocalMapReduceRunner:
    """
    Simulates MapReduce locally using Unix pipes for development and testing.
    Reads CSVs from local directory, runs mapper and reducer scripts.
    """

    def run(self, input_dir: str, output_path: str) -> int:
        input_files = list(Path(input_dir).glob("**/*.csv"))
        if not input_files:
            logger.error(f"No CSV files found in {input_dir}")
            return 1

        logger.info(f"Running local MapReduce simulation on {len(input_files)} files...")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / "part-00000.tsv"

        cat_cmd = ["cat"] + [str(f) for f in input_files]
        sort_cmd = ["sort", "-k1,1"]
        mapper_cmd = ["python3", str(MAPPER_PATH)]
        reducer_cmd = ["python3", str(REDUCER_PATH)]

        logger.info("Running: cat | mapper | sort | reducer")

        try:
            with open(output_file, "w") as out_f:
                p_cat = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
                p_map = subprocess.Popen(mapper_cmd, stdin=p_cat.stdout, stdout=subprocess.PIPE)
                p_cat.stdout.close()
                p_sort = subprocess.Popen(sort_cmd, stdin=p_map.stdout, stdout=subprocess.PIPE)
                p_map.stdout.close()
                p_red = subprocess.Popen(reducer_cmd, stdin=p_sort.stdout, stdout=out_f)
                p_sort.stdout.close()
                p_red.wait()

            logger.info(f"Local MapReduce complete. Output: {output_file}")
            return 0
        except Exception as e:
            logger.error(f"Local MapReduce failed: {e}")
            return 1


def main():
    parser = argparse.ArgumentParser(description="Run MapReduce churn feature engineering job")
    parser.add_argument("--input", required=True, help="Input path (HDFS/S3 or local directory)")
    parser.add_argument("--output", required=True, help="Output path (HDFS/S3 or local directory)")
    parser.add_argument("--mode", choices=["local", "hadoop"], default="local", help="Execution mode")
    parser.add_argument("--num-reducers", type=int, default=4, help="Number of reducers (Hadoop mode)")
    args = parser.parse_args()

    if args.mode == "local":
        runner = LocalMapReduceRunner()
        exit_code = runner.run(args.input, args.output)
    else:
        runner = HadoopStreamingRunner()
        exit_code = runner.run(args.input, args.output, args.num_reducers)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
