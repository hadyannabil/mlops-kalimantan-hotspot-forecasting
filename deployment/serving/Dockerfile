FROM kalimantan-hotspot-mlflow-serving:lk10

RUN python -m pip install --no-cache-dir \
    prometheus-client==0.22.1

COPY serving.py /opt/lk11/serving.py
COPY drift_metrics.py /opt/lk11/drift_metrics.py
COPY drift_baseline.json /opt/lk11/drift_baseline.json

EXPOSE 8080

ENTRYPOINT ["python", "/opt/lk11/serving.py"]
