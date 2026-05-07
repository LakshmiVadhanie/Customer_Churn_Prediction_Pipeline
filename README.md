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

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.88 |
| Precision | 0.83 |
| Recall | 0.79 |
| F1-Score | 0.81 |
| Segments | 12 |


