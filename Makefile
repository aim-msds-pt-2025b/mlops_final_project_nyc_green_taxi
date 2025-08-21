SHELL := /bin/bash

.PHONY: data transform train validate drift api airflow-init

data:
	python -m src.data.get_data

transform:
	python -m src.features.transform

train:
	python -m src.models.train

validate:
	python -m src.models.validate

drift:
	python -m src.data.simulate_drift && python -m src.monitoring.generate_drift

api:
	uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

airflow-init:
	docker compose run --rm airflow-webserver airflow db init && \
	docker compose run --rm airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
