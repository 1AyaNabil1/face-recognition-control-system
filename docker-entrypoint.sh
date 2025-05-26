#!/bin/bash
set -e

# Function to wait for a service
wait_for_service() {
    local host="$1"
    local port="$2"
    local service="$3"
    local timeout="${4:-30}"

    echo "Waiting for $service to be ready..."
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port"; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... $i/$timeout"
        sleep 1
    done
    echo >&2 "$service is not available"
    return 1
}

# Check if we're running in debug mode
if [ "${DEBUG:-false}" = "true" ]; then
    echo "Running in debug mode"
    export PYTHONPATH=/app
    export LOG_LEVEL=DEBUG
fi

# Wait for Redis if configured
if [ -n "$REDIS_HOST" ]; then
    wait_for_service "$REDIS_HOST" "${REDIS_PORT:-6379}" "Redis"
fi

# Wait for PostgreSQL if configured
if [ -n "$DATABASE_URL" ]; then
    DB_HOST=$(echo $DATABASE_URL | awk -F[@/] '{print $4}')
    DB_PORT=$(echo $DATABASE_URL | awk -F[@:/] '{print $5}')
    wait_for_service "$DB_HOST" "${DB_PORT:-5432}" "PostgreSQL"
fi

# Download models if needed
if [ "${DOWNLOAD_MODELS:-true}" = "true" ]; then
    echo "Checking and downloading models..."
    python -c "from app.models.model_manager import ModelManager; ModelManager().download_all()"
fi

# Run database migrations if needed
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start Prometheus exporter if enabled
if [ "${ENABLE_METRICS:-true}" = "true" ]; then
    echo "Starting Prometheus exporter..."
    prometheus_multiproc_dir=/tmp/prom_metrics
    mkdir -p "$prometheus_multiproc_dir"
    export prometheus_multiproc_dir
fi

# Set number of workers based on CPU cores
if [ -z "${WORKERS}" ]; then
    WORKERS=$((2 * $(nproc) + 1))
fi

# Execute the main command
exec "$@" 