-- create_tables.hql
-- Hive DDL for customer churn prediction pipeline
-- Creates raw log table and churn-ready feature table
-- Run: hive -f src/hive/create_tables.hql

CREATE DATABASE IF NOT EXISTS churn_db
COMMENT 'Customer churn prediction database'
LOCATION 's3://YOUR_BUCKET/hive/warehouse/';

USE churn_db;

-- Raw customer logs table (partitioned by date for scan efficiency)
DROP TABLE IF EXISTS customer_logs;

CREATE EXTERNAL TABLE customer_logs (
    customer_id         STRING          COMMENT 'Unique customer identifier',
    segment             STRING          COMMENT 'Customer segment (enterprise, smb, etc.)',
    plan                STRING          COMMENT 'Subscription plan',
    region              STRING          COMMENT 'Geographic region',
    device              STRING          COMMENT 'Primary device type',
    sessions            INT             COMMENT 'Number of sessions',
    total_events        INT             COMMENT 'Total event count',
    api_calls           INT             COMMENT 'API call count',
    support_tickets     INT             COMMENT 'Support ticket count',
    feature_adoption_score DOUBLE       COMMENT 'Feature adoption 0.0-1.0',
    nps_score           DOUBLE          COMMENT 'NPS score (-1 if no response)',
    days_since_signup   INT             COMMENT 'Days since customer signup',
    monthly_contract_value DOUBLE       COMMENT 'MCV in USD',
    last_login_days_ago INT             COMMENT 'Days since last login',
    churn_label         INT             COMMENT '1 = churned within 30 days',
    date                STRING          COMMENT 'Log date YYYY-MM-DD'
)
PARTITIONED BY (log_date STRING)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION 's3://YOUR_BUCKET/raw/customer_logs/'
TBLPROPERTIES (
    'skip.header.line.count'='1',
    'serialization.null.format'=''
);

-- Repair partitions after S3 upload
MSCK REPAIR TABLE customer_logs;


-- Aggregated churn feature table (output from MapReduce)
DROP TABLE IF EXISTS churn_features;

CREATE EXTERNAL TABLE churn_features (
    customer_id                 STRING,
    segment                     STRING,
    plan                        STRING,
    region                      STRING,
    primary_device              STRING,
    total_sessions              INT,
    total_events                INT,
    total_api_calls             INT,
    total_support_tickets       INT,
    avg_feature_adoption_score  DOUBLE,
    avg_sessions_per_day        DOUBLE,
    avg_events_per_session      DOUBLE,
    avg_api_calls_per_day       DOUBLE,
    max_last_login_days_ago     INT,
    nps_score_avg               DOUBLE,
    nps_response_rate           DOUBLE,
    days_since_signup           INT,
    monthly_contract_value      DOUBLE,
    activity_trend              DOUBLE,
    support_escalation_rate     DOUBLE,
    engagement_score            DOUBLE,
    churn_label                 INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION 's3://YOUR_BUCKET/processed/features/'
TBLPROPERTIES ('skip.header.line.count'='1');


-- Churn summary by segment (for Tableau export)
DROP VIEW IF EXISTS churn_by_segment;

CREATE VIEW churn_by_segment AS
SELECT
    segment,
    plan,
    region,
    COUNT(*)                            AS total_customers,
    SUM(churn_label)                    AS churned_customers,
    ROUND(AVG(churn_label) * 100, 2)   AS churn_rate_pct,
    ROUND(AVG(engagement_score), 4)    AS avg_engagement,
    ROUND(AVG(monthly_contract_value), 2) AS avg_mcv,
    ROUND(AVG(avg_sessions_per_day), 2)  AS avg_sessions_daily,
    ROUND(AVG(total_support_tickets), 2) AS avg_support_tickets,
    ROUND(AVG(nps_score_avg), 2)        AS avg_nps
FROM churn_features
GROUP BY segment, plan, region;


-- High-risk customer identification query
CREATE VIEW high_risk_customers AS
SELECT
    customer_id,
    segment,
    plan,
    engagement_score,
    churn_label,
    monthly_contract_value,
    max_last_login_days_ago,
    total_support_tickets,
    activity_trend
FROM churn_features
WHERE engagement_score < 0.25
  AND max_last_login_days_ago >= 14
ORDER BY engagement_score ASC;
