# syntax=docker/dockerfile:1
# Build context: repository root.   docker build -f services/api/Dockerfile -t chessvision-api .
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY services/api/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.11-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    WEIGHTS_DIR=/models DB_URL=sqlite:////data/history.db STOCKFISH_PATH=/usr/games/stockfish
# Stockfish (GPL-3.0) is installed as a separate binary and driven over UCI.
RUN apt-get update && apt-get install -y --no-install-recommends stockfish libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home app && mkdir /data /models && chown app /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY ml/__init__.py ml/__init__.py
COPY ml/chessvision ml/chessvision
COPY services/__init__.py services/__init__.py
COPY services/api services/api
USER app
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "services.api.app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
