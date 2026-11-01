FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependency model dan API
COPY docker_assets/model/requirements.txt /tmp/model-requirements.txt

RUN python -m pip install --no-cache-dir \
    -r /tmp/model-requirements.txt \
    "mlflow==3.17.1" \
    "numpy==2.5.2" \
    "pandas==3.0.5" \
    "scikit-learn==1.9.0" \
    "skops" \
    "Flask>=3.1,<4" \
    "gunicorn>=23,<26"

# Salin API dan model Production v1
COPY api/app.py /app/api/app.py
COPY docker_assets/model/ /app/docker_assets/model/

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "1", "--timeout", "120", \
     "api.app:app"]
