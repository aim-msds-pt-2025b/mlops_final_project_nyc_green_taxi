FROM apache/airflow:2.9.3-python3.11
USER root
COPY docker/airflow-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
USER airflow
