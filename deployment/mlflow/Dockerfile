
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

RUN python -m pip install --no-cache-dir \
    "mlflow==3.17.1" \
    "psycopg2-binary>=2.9,<3"

EXPOSE 5000