# Customer Churn Prediction Pipeline

End-to-end big data pipeline for predicting 30-day customer churn using Hadoop, Hive, MapReduce, PySpark, AWS S3, and Tableau-compatible exports.

## Architecture

```
AWS S3 (Raw Logs)
      |
      v
MapReduce (Hadoop) -- Aggregation & Feature Engineering
      |
      v
Hive (Feature Tables) -- Structured Storage
      |
      v
PySpark (GBT Model) -- 88% AUC, 12 Segments
      |
      v
Dashboard (Flask + Chart.js) -- Real-time Inference UI
      |
      v
Tableau Export -- CSV/JSON for BI Visualization
```

## Tech Stack

| Layer | Technology |
|---|---|
| Storage | AWS S3 |
| Distributed Compute | Hadoop 3.x, MapReduce |
| Data Warehouse | Apache Hive |
| ML Engine | PySpark MLlib (GBT) |
| Orchestration | Python 3.10+ |
| Dashboard | Flask, Chart.js |
| Visualization | Tableau-compatible exports |

## Prerequisites

- Python 3.10+
- Java 8+ (for Hadoop/Hive)
- Apache Hadoop 3.3+
- Apache Hive 3.1+
- Apache Spark 3.4+
- AWS CLI configured
- pip packages in requirements.txt

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your S3 bucket, Hadoop paths, etc.
```

### 3. Generate sample data (local dev)

```bash
python src/aws/s3_data_generator.py --records 2000000 --output data/sample/
```

### 4. Upload to S3

```bash
python src/aws/s3_uploader.py --local data/sample/ --bucket YOUR_BUCKET
```

### 5. Run MapReduce aggregation

```bash
python src/mapreduce/runner.py --input s3://YOUR_BUCKET/logs/ --output s3://YOUR_BUCKET/features/
```

### 6. Create Hive tables

```bash
hive -f src/hive/create_tables.hql
```

### 7. Train PySpark model

```bash
spark-submit src/spark/train_model.py --input s3://YOUR_BUCKET/features/ --model-output models/
```

### 8. Launch dashboard

```bash
python dashboard/app.py
```

Open http://localhost:5000

## Running Full Pipeline

```bash
python src/pipeline.py --env production --s3-bucket YOUR_BUCKET
```

## Local Mode (No Hadoop/S3 Required)

```bash
python src/pipeline.py --env local --records 50000
```

## Project Structure

```
churn-prediction/
    config/             # YAML configuration
    data/               # Sample and intermediate data
    dashboard/          # Flask web application
    notebooks/          # Jupyter EDA notebooks
    src/
        aws/            # S3 upload/download utilities
        hive/           # HQL DDL and queries
        mapreduce/      # Mapper, reducer, runner
        spark/          # PySpark training and inference
    tests/              # Unit and integration tests
```

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.88 |
| Precision | 0.83 |
| Recall | 0.79 |
| F1-Score | 0.81 |
| Segments | 12 |

## License

MIT
