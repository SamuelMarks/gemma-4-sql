FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies, including git for maxtext
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app
# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
# Ensure the virtual environment is in the PATH
ENV PATH="/opt/venv/bin:$PATH"

COPY . .
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "gemma_4_sql.cli"]
