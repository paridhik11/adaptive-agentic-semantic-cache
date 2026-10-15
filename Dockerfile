FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY deploy/requirements-gateway.txt /app/deploy/requirements-gateway.txt
RUN pip install --no-cache-dir -r deploy/requirements-gateway.txt \
    && groupadd --gid 10001 cacheapp \
    && useradd --uid 10001 --gid cacheapp --no-create-home cacheapp \
    && mkdir /state && chown cacheapp:cacheapp /state
COPY src /app/src
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["python", "-m", "uvicorn", "src.api.gateway:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "64", "--no-access-log"]
