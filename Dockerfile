# Multi-stage Docker build for NYC School Analyzer
# Production-ready containerization with security and performance optimizations

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL maintainer="NYC School Analytics Team <analytics@nycschools.edu>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="nyc-school-analyzer" \
      org.label-schema.description="Production-ready NYC High School Directory Analysis Tool" \
      org.label-schema.url="https://github.com/nyc-schools/nyc-school-analyzer" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nyc-schools/nyc-school-analyzer" \
      org.label-schema.vendor="NYC Department of Education" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r nyc-schools && \
    useradd -r -g nyc-schools -d /app -s /bin/bash nyc-schools

# Set work directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /build/src ./src
COPY --from=builder /build/config ./config
COPY --from=builder /build/setup.py .
COPY --from=builder /build/README.md .

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/logs /app/charts /app/reports && \
    chown -R nyc-schools:nyc-schools /app

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NYC_SCHOOLS_LOG_LEVEL=INFO \
    NYC_SCHOOLS_OUTPUT_PATH=/app/outputs \
    MPLBACKEND=Agg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import nyc_school_analyzer; print('OK')" || exit 1

# Switch to non-root user
USER nyc-schools

# Expose volume mount points
VOLUME ["/app/data", "/app/outputs", "/app/logs"]

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["--help"]