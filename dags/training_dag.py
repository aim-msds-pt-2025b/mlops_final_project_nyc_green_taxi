# Callables import our project modules mounted at /opt/airflow/src
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add src to Python path
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

from src.data.get_data import main as get_data_main
from src.features.transform import main as transform_main
from src.models.train import main as train_main
from src.models.validate import main as validate_main

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="training_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="Daily training pipeline: ingest -> transform -> train -> validate",
) as dag:
    t_ingest = PythonOperator(task_id="ingest", python_callable=get_data_main)
    t_transform = PythonOperator(task_id="transform", python_callable=transform_main)
    t_train = PythonOperator(task_id="train", python_callable=train_main)
    t_validate = PythonOperator(task_id="validate", python_callable=validate_main)

    t_ingest >> t_transform >> t_train >> t_validate
