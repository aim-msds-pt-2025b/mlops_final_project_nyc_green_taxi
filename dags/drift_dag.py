import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to Python path
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

from src.data.simulate_drift import main as simulate_drift_main
from src.monitoring.generate_drift import main as drift_report_main

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="drift_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="Hourly drift simulation & reporting with Evidently",
) as dag:
    t_simulate = PythonOperator(task_id="simulate_current", python_callable=simulate_drift_main)
    t_drift = PythonOperator(task_id="generate_drift_reports", python_callable=drift_report_main)

    t_simulate >> t_drift
