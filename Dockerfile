FROM python:3.11-slim AS builder

WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml poetry.lock* ./

# Configure poetry to create the virtual environment in the project root (.venv)
RUN pip install -r requirements.txt
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root

FROM python:3.11-slim AS runner

WORKDIR /app

# Install supervisor and clean apt cache to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code and configuration
COPY camera_traps/ ./camera_traps/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Add the virtual environment to PATH so Python binaries are directly available
ENV PATH="/app/.venv/bin:$PATH"

# Expose ports (e.g. 8000 backend, 8501 frontend)
EXPOSE 8000 8501

# Run supervisor to manage multiple processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
