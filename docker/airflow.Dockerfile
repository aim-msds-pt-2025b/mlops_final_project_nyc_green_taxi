FROM apache/airflow:2.9.3-python3.11
USER root
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.4.2 mlflow==2.14.1 evidently==0.4.29 shap==0.45.1 pyyaml==6.0.1 requests==2.32.3 joblib==1.4.2 matplotlib==3.8.4
USER airflow
