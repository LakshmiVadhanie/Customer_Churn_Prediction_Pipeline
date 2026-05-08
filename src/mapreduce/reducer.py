#!/usr/bin/env python3
"""
reducer.py

Hadoop Streaming MapReduce Reducer for customer churn feature aggregation.
Receives sorted (customer_id, features) pairs from stdin, aggregates
30-day window statistics per customer, and emits churn-ready feature rows.

Output format: TSV with aggregated feature vector per customer.
"""

import sys
import json
import logging
from itertools import groupby

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

OUTPUT_HEADER = "\t".join([
    "customer_id",
    "segment",
    "plan",
    "region",
    "primary_device",
    "total_sessions",
    "total_events",
    "total_api_calls",
    "total_support_tickets",
    "avg_feature_adoption_score",
    "avg_sessions_per_day",
    "avg_events_per_session",
    "avg_api_calls_per_day",
    "max_last_login_days_ago",
    "nps_score_avg",
    "nps_response_rate",
    "days_since_signup",
    "monthly_contract_value",
    "activity_trend",
    "support_escalation_rate",
    "engagement_score",
    "churn_label",
])


def compute_trend(sessions_list: list) -> float:
    n = len(sessions_list)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(sessions_list) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(sessions_list))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0


def aggregate_customer(customer_id: str, records: list) -> str:
    n = len(records)
    if n == 0:
        return None

    segment = records[0].get("segment", "unknown")
    plan = records[0].get("plan", "unknown")
    region = records[0].get("region", "unknown")
    days_since_signup = records[0].get("days_since_signup", 0)
    monthly_contract_value = records[0].get("monthly_contract_value", 0.0)
    churn_label = records[0].get("churn_label", 0)

    device_counts = {}
    for r in records:
        d = r.get("device", "web")
        device_counts[d] = device_counts.get(d, 0) + 1
    primary_device = max(device_counts, key=device_counts.get)

    total_sessions = sum(r.get("sessions", 0) for r in records)
    total_events = sum(r.get("total_events", 0) for r in records)
    total_api_calls = sum(r.get("api_calls", 0) for r in records)
    total_support_tickets = sum(r.get("support_tickets", 0) for r in records)

    avg_feature_adoption = sum(r.get("feature_adoption_score", 0.0) for r in records) / n
    avg_sessions_per_day = total_sessions / n
    avg_events_per_session = total_events / total_sessions if total_sessions > 0 else 0
    avg_api_calls_per_day = total_api_calls / n
    max_last_login_days_ago = max(r.get("last_login_days_ago", 0) for r in records)

    nps_scores = [r.get("nps_score", -1) for r in records if r.get("nps_score", -1) >= 0]
    nps_avg = sum(nps_scores) / len(nps_scores) if nps_scores else -1
    nps_response_rate = len(nps_scores) / n

    sessions_ordered = [r.get("sessions", 0) for r in records]
    activity_trend = compute_trend(sessions_ordered)

    support_escalation_rate = total_support_tickets / n

    engagement_score = (
        0.30 * min(avg_sessions_per_day / 5.0, 1.0) +
        0.25 * avg_feature_adoption +
        0.20 * min(avg_api_calls_per_day / 20.0, 1.0) +
        0.15 * (1.0 - min(max_last_login_days_ago / 30.0, 1.0)) +
        0.10 * (max(activity_trend, 0) / 5.0)
    )

    row = "\t".join([
        customer_id,
        segment,
        plan,
        region,
        primary_device,
        str(total_sessions),
        str(total_events),
        str(total_api_calls),
        str(total_support_tickets),
        f"{avg_feature_adoption:.4f}",
        f"{avg_sessions_per_day:.4f}",
        f"{avg_events_per_session:.4f}",
        f"{avg_api_calls_per_day:.4f}",
        str(max_last_login_days_ago),
        f"{nps_avg:.4f}",
        f"{nps_response_rate:.4f}",
        str(days_since_signup),
        f"{monthly_contract_value:.2f}",
        f"{activity_trend:.6f}",
        f"{support_escalation_rate:.4f}",
        f"{engagement_score:.4f}",
        str(churn_label),
    ])
    return row


def main():
    print(OUTPUT_HEADER)

    def read_pairs():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                sys.stderr.write(f"Malformed reducer input: {line}\n")
                continue
            customer_id, json_str = parts
            try:
                record = json.loads(json_str)
                yield customer_id, record
            except json.JSONDecodeError as e:
                sys.stderr.write(f"JSON parse error: {e}\n")
                continue

    for customer_id, group in groupby(read_pairs(), key=lambda x: x[0]):
        records = [r for _, r in group]
        row = aggregate_customer(customer_id, records)
        if row:
            print(row)


if __name__ == "__main__":
    main()
