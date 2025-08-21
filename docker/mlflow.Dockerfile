FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.14.1 psycopg2-binary==2.9.9
ENV BACKEND_STORE_URI=sqlite:///mlflow.db
ENV ARTIFACT_ROOT=/mlruns
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "${BACKEND_STORE_URI}", "--default-artifact-root", "${ARTIFACT_ROOT}", "--host", "0.0.0.0", "--port", "5000"]
