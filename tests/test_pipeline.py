"""
test_pipeline.py

Unit tests for mapper, reducer, inference, and data generation modules.
Run: python -m pytest tests/ -v
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMapper(unittest.TestCase):
    def _run_mapper(self, csv_lines: list) -> list:
        from mapreduce.mapper import process_record, EXPECTED_FIELDS
        outputs = []
        import csv as csv_mod
        from io import StringIO
        reader = csv_mod.reader(StringIO("\n".join(csv_lines)))
        header = None
        for i, row in enumerate(reader):
            if i == 0:
                header = [c.strip() for c in row]
                if header[0] == "customer_id":
                    continue
            if header is None:
                header = EXPECTED_FIELDS
            record = dict(zip(header, row))
            captured = []
            import builtins
            original_print = builtins.print

            def mock_print(*args, **kwargs):
                captured.append("\t".join(str(a) for a in args))

            builtins.print = mock_print
            process_record(record)
            builtins.print = original_print
            outputs.extend(captured)
        return outputs

    def test_valid_record_emits_key_value(self):
        header = "customer_id,date,segment,plan,sessions,total_events,api_calls,support_tickets,feature_adoption_score,nps_score,days_since_signup,monthly_contract_value,last_login_days_ago,churn_label,region,device"
        row = "CUST0001234,2024-01-15,enterprise,pro,12,85,300,0,0.75,9,365,299.0,1,0,us-east,web"
        outputs = self._run_mapper([header, row])
        self.assertEqual(len(outputs), 1)
        key, val_json = outputs[0].split("\t", 1)
        self.assertEqual(key, "CUST0001234")
        val = json.loads(val_json)
        self.assertEqual(val["segment"], "enterprise")
        self.assertEqual(val["sessions"], 12)
        self.assertEqual(val["churn_label"], 0)

    def test_empty_customer_id_skipped(self):
        header = "customer_id,date,segment,plan,sessions,total_events,api_calls,support_tickets,feature_adoption_score,nps_score,days_since_signup,monthly_contract_value,last_login_days_ago,churn_label,region,device"
        row = ",2024-01-15,enterprise,pro,12,85,300,0,0.75,9,365,299.0,1,0,us-east,web"
        outputs = self._run_mapper([header, row])
        self.assertEqual(len(outputs), 0)

    def test_numeric_fields_safe_parsed(self):
        header = "customer_id,date,segment,plan,sessions,total_events,api_calls,support_tickets,feature_adoption_score,nps_score,days_since_signup,monthly_contract_value,last_login_days_ago,churn_label,region,device"
        row = "CUST9999,2024-01-01,smb,basic,BAD,BROKEN,0,0,0.5,-1,100,99.0,3,0,eu-west,mobile_ios"
        outputs = self._run_mapper([header, row])
        self.assertEqual(len(outputs), 1)
        val = json.loads(outputs[0].split("\t", 1)[1])
        self.assertEqual(val["sessions"], 0)
        self.assertEqual(val["total_events"], 0)


class TestReducer(unittest.TestCase):
    def test_aggregate_single_record(self):
        from mapreduce.reducer import aggregate_customer
        records = [{
            "segment": "enterprise", "plan": "pro", "region": "us-east",
            "device": "web", "sessions": 10, "total_events": 80,
            "api_calls": 200, "support_tickets": 0, "feature_adoption_score": 0.8,
            "nps_score": 9, "days_since_signup": 400, "monthly_contract_value": 500.0,
            "last_login_days_ago": 2, "churn_label": 0
        }]
        row = aggregate_customer("CUST0001", records)
        self.assertIsNotNone(row)
        cols = row.split("\t")
        self.assertEqual(cols[0], "CUST0001")
        self.assertEqual(cols[1], "enterprise")

    def test_aggregate_multiple_records(self):
        from mapreduce.reducer import aggregate_customer
        records = [
            {"segment": "smb", "plan": "basic", "region": "eu-west", "device": "web",
             "sessions": 5, "total_events": 40, "api_calls": 100, "support_tickets": 1,
             "feature_adoption_score": 0.5, "nps_score": 7, "days_since_signup": 200,
             "monthly_contract_value": 99.0, "last_login_days_ago": 3, "churn_label": 0},
            {"segment": "smb", "plan": "basic", "region": "eu-west", "device": "web",
             "sessions": 0, "total_events": 0, "api_calls": 0, "support_tickets": 2,
             "feature_adoption_score": 0.2, "nps_score": -1, "days_since_signup": 200,
             "monthly_contract_value": 99.0, "last_login_days_ago": 10, "churn_label": 1},
        ]
        row = aggregate_customer("CUST0002", records)
        cols = row.split("\t")
        total_sessions = int(cols[5])
        self.assertEqual(total_sessions, 5)
        total_tickets = int(cols[8])
        self.assertEqual(total_tickets, 3)

    def test_compute_trend_positive(self):
        from mapreduce.reducer import compute_trend
        trend = compute_trend([1, 2, 3, 4, 5])
        self.assertGreater(trend, 0)

    def test_compute_trend_negative(self):
        from mapreduce.reducer import compute_trend
        trend = compute_trend([5, 4, 3, 2, 1])
        self.assertLess(trend, 0)

    def test_compute_trend_flat(self):
        from mapreduce.reducer import compute_trend
        trend = compute_trend([3, 3, 3, 3])
        self.assertAlmostEqual(trend, 0.0)


class TestInference(unittest.TestCase):
    def test_simulated_prediction(self):
        from spark.inference import ChurnPredictor
        predictor = ChurnPredictor(model_path="nonexistent_model_path")
        features = {
            "engagement_score": 0.2,
            "max_last_login_days_ago": 20,
            "total_support_tickets": 3,
            "activity_trend": -1.0,
        }
        result = predictor.predict_single(features)
        self.assertIn("churn_probability", result)
        self.assertIn("risk_tier", result)
        self.assertIn("top_features", result)
        self.assertGreaterEqual(result["churn_probability"], 0.0)
        self.assertLessEqual(result["churn_probability"], 1.0)
        self.assertIn(result["risk_tier"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"])

    def test_low_risk_customer(self):
        from spark.inference import ChurnPredictor
        predictor = ChurnPredictor(model_path="nonexistent_model_path")
        features = {
            "engagement_score": 0.95,
            "max_last_login_days_ago": 0,
            "total_support_tickets": 0,
            "activity_trend": 2.0,
        }
        result = predictor.predict_single(features)
        self.assertIn(result["risk_tier"], ["LOW", "MEDIUM"])

    def test_get_segment_stats_returns_12(self):
        from spark.inference import ChurnPredictor
        predictor = ChurnPredictor(model_path="nonexistent_model_path")
        stats = predictor.get_segment_stats()
        self.assertEqual(len(stats), 12)

    def test_get_risk_tier(self):
        from spark.inference import get_risk_tier
        self.assertEqual(get_risk_tier(0.90), "CRITICAL")
        self.assertEqual(get_risk_tier(0.60), "HIGH")
        self.assertEqual(get_risk_tier(0.40), "MEDIUM")
        self.assertEqual(get_risk_tier(0.10), "LOW")


class TestDataGenerator(unittest.TestCase):
    def test_generate_small_dataset(self):
        import tempfile
        import csv
        from aws.s3_data_generator import generate_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            count = generate_dataset(num_records=3000, output_dir=tmpdir, churn_rate=0.2)
            self.assertGreater(count, 0)
            csvfiles = list(Path(tmpdir).glob("*.csv"))
            self.assertGreater(len(csvfiles), 0)
            with open(csvfiles[0]) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertGreater(len(rows), 0)
                self.assertIn("customer_id", rows[0])
                self.assertIn("churn_label", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
