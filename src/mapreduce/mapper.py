#!/usr/bin/env python3
"""
mapper.py

Hadoop Streaming MapReduce Mapper for customer churn feature engineering.
Reads raw CSV customer log records from stdin, emits (customer_id, features) pairs.

Usage (Hadoop Streaming):
    hadoop jar hadoop-streaming.jar \
        -input s3://bucket/raw/customer_logs/ \
        -output s3://bucket/processed/features/ \
        -mapper "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -file mapper.py \
        -file reducer.py
"""

import sys
import csv
import json
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

EXPECTED_FIELDS = [
    "customer_id", "date", "segment", "plan", "sessions", "total_events",
    "api_calls", "support_tickets", "feature_adoption_score", "nps_score",
    "days_since_signup", "monthly_contract_value", "last_login_days_ago",
    "churn_label", "region", "device"
]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def emit(key: str, value: dict):
    print(f"{key}\t{json.dumps(value)}")


def process_record(record: dict):
    customer_id = record.get("customer_id", "").strip()
    if not customer_id:
        return

    features = {
        "date": record.get("date", ""),
        "segment": record.get("segment", "unknown"),
        "plan": record.get("plan", "unknown"),
        "region": record.get("region", "unknown"),
        "device": record.get("device", "unknown"),
        "sessions": safe_int(record.get("sessions", 0)),
        "total_events": safe_int(record.get("total_events", 0)),
        "api_calls": safe_int(record.get("api_calls", 0)),
        "support_tickets": safe_int(record.get("support_tickets", 0)),
        "feature_adoption_score": safe_float(record.get("feature_adoption_score", 0.0)),
        "nps_score": safe_float(record.get("nps_score", -1)),
        "days_since_signup": safe_int(record.get("days_since_signup", 0)),
        "monthly_contract_value": safe_float(record.get("monthly_contract_value", 0.0)),
        "last_login_days_ago": safe_int(record.get("last_login_days_ago", 0)),
        "churn_label": safe_int(record.get("churn_label", 0)),
    }

    emit(customer_id, features)


def main():
    header = None
    reader = csv.reader(sys.stdin)

    for i, row in enumerate(reader):
        if i == 0:
            header = [col.strip() for col in row]
            if header[0] == "customer_id":
                continue

        if header is None:
            header = EXPECTED_FIELDS

        try:
            if len(row) != len(header):
                sys.stderr.write(f"Skipping malformed row {i}: {row}\n")
                continue

            record = dict(zip(header, row))
            process_record(record)

        except Exception as e:
            sys.stderr.write(f"Mapper error on row {i}: {e}\n")
            continue


if __name__ == "__main__":
    main()
