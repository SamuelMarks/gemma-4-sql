FROM python:3.12-alpine AS builder

WORKDIR /app
COPY requirements.txt .

# Install necessary build dependencies for Alpine, including git for maxtext
RUN apk add --no-cache git build-base linux-headers python3-dev && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-alpine

WORKDIR /app
# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
# Ensure the virtual environment is in the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Some Python libraries require basic standard library dependencies at runtime even on Alpine
RUN apk add --no-cache libstdc++ 

COPY . .
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "gemma_4_sql.cli"]
