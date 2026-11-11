# syntax=docker/dockerfile:1
# Build context: repository root.   docker build -f services/dashboard/Dockerfile -t chessvision-dashboard .
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY services/dashboard/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://api:8000 SAMPLES_DIR=/app/samples
RUN useradd --uid 10001 --create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY services/__init__.py services/__init__.py
COPY services/dashboard services/dashboard
COPY samples samples
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"
CMD ["streamlit", "run", "services/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
