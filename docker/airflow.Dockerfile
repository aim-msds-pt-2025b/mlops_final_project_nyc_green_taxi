FROM apache/airflow:2.9.3-python3.11
USER airflow

# Copy requirements and install all dependencies needed for DAGs
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH="/opt/airflow/src:/opt/airflow:${PYTHONPATH}"
