from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
sys.path.append('/opt/airflow/src')

from src.deployment.promote import main as promote_main

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="deployment_dag",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="Register & promote model to Production if thresholds pass; trigger API reload",
) as dag:

    t_promote = PythonOperator(task_id="promote_and_reload", python_callable=promote_main)

    t_promote
