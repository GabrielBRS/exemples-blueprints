# ---- builder: resolve deps com uv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml ./
# instale o extra do adapter que for usar: .[langgraph] | .[autogen] | .[all]
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python ".[all]"

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN useradd -r -u 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app
ENV PATH="/opt/venv/bin:$PATH" HOST=0.0.0.0 PORT=8080 LOG_LEVEL=INFO ORCHESTRATOR=echo
USER appuser
EXPOSE 8080
# Probes do k3s: liveness -> /health/live, readiness -> /health/ready
ENTRYPOINT ["python", "-m", "app.main"]
