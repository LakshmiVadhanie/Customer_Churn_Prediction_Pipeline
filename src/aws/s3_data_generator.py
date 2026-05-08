"""
s3_data_generator.py

Generates synthetic customer log data simulating 2M+ records
matching production schema used in the MapReduce pipeline.
"""

import csv
import random
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEGMENTS = [
    "enterprise", "smb", "startup", "consumer_free", "consumer_paid",
    "trial", "government", "education", "healthcare", "finance",
    "retail", "technology"
]

EVENTS = [
    "login", "logout", "page_view", "feature_use", "support_ticket",
    "billing_view", "export", "api_call", "settings_change", "invite_sent"
]

PLANS = ["free", "basic", "pro", "enterprise"]


def generate_customer_id():
    return f"CUST{random.randint(100000, 9999999):07d}"


def generate_log_record(customer_id: str, date: datetime, is_churn: bool) -> dict:
    segment = random.choice(SEGMENTS)
    plan = random.choice(PLANS)

    base_activity = 3 if not is_churn else 0.5
    sessions = max(0, int(np.random.poisson(base_activity)))
    events = max(0, int(np.random.poisson(sessions * 4.5)))
    api_calls = max(0, int(np.random.poisson(sessions * 10)))

    support_tickets = 0
    if is_churn and random.random() < 0.4:
        support_tickets = random.randint(1, 5)

    days_since_signup = random.randint(1, 1825)
    contract_value = {
        "free": 0,
        "basic": random.uniform(29, 49),
        "pro": random.uniform(99, 299),
        "enterprise": random.uniform(999, 9999)
    }[plan]

    return {
        "customer_id": customer_id,
        "date": date.strftime("%Y-%m-%d"),
        "segment": segment,
        "plan": plan,
        "sessions": sessions,
        "total_events": events,
        "api_calls": api_calls,
        "support_tickets": support_tickets,
        "feature_adoption_score": round(random.uniform(0.1, 1.0) if not is_churn else random.uniform(0.0, 0.4), 4),
        "nps_score": random.randint(0, 10) if random.random() < 0.15 else -1,
        "days_since_signup": days_since_signup,
        "monthly_contract_value": round(contract_value, 2),
        "last_login_days_ago": random.randint(1, 30) if not is_churn else random.randint(15, 90),
        "churn_label": int(is_churn),
        "region": random.choice(["us-east", "us-west", "eu-west", "ap-south", "ap-east"]),
        "device": random.choice(["web", "mobile_ios", "mobile_android", "api"]),
    }


def generate_dataset(num_records: int, output_dir: str, churn_rate: float = 0.18):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {num_records:,} customer log records...")

    num_customers = num_records // 30
    churn_customers = int(num_customers * churn_rate)
    active_customers = num_customers - churn_customers

    customers = []
    for _ in range(active_customers):
        customers.append((generate_customer_id(), False))
    for _ in range(churn_customers):
        customers.append((generate_customer_id(), True))

    random.shuffle(customers)

    base_date = datetime(2024, 1, 1)
    records_written = 0
    file_index = 0
    batch_size = 500000

    fieldnames = [
        "customer_id", "date", "segment", "plan", "sessions", "total_events",
        "api_calls", "support_tickets", "feature_adoption_score", "nps_score",
        "days_since_signup", "monthly_contract_value", "last_login_days_ago",
        "churn_label", "region", "device"
    ]

    current_file = None
    writer = None

    for cust_id, is_churn in customers:
        for day_offset in range(30):
            if records_written % batch_size == 0:
                if current_file:
                    current_file.close()
                    logger.info(f"Written {records_written:,} records...")
                fname = output_path / f"customer_logs_part_{file_index:04d}.csv"
                current_file = open(fname, "w", newline="")
                writer = csv.DictWriter(current_file, fieldnames=fieldnames)
                writer.writeheader()
                file_index += 1

            record_date = base_date + timedelta(days=day_offset)
            record = generate_log_record(cust_id, record_date, is_churn)
            writer.writerow(record)
            records_written += 1

    if current_file:
        current_file.close()

    logger.info(f"Dataset generation complete. Total records: {records_written:,}")
    logger.info(f"Output files: {output_path}")
    return records_written


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer churn log data")
    parser.add_argument("--records", type=int, default=2_000_000, help="Number of log records to generate")
    parser.add_argument("--output", type=str, default="data/sample/", help="Output directory")
    parser.add_argument("--churn-rate", type=float, default=0.18, help="Proportion of churned customers")
    args = parser.parse_args()

    generate_dataset(args.records, args.output, args.churn_rate)


if __name__ == "__main__":
    main()
