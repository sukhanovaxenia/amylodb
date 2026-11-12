# amylodb core image: curated build + reconciliation + features + graph export.
# Mining extras (torch/transformers/spacy) are intentionally NOT installed here
# to keep the image small; build the mining image separately if needed
# (pip install ".[mining]") on a host with model/network access.
FROM python:3.11-slim

LABEL org.opencontainers.image.title="amylodb" \
      org.opencontainers.image.description="Unified amyloid database (curated + mined)" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml]"

# Default: show CLI help. Mount data and pass a config at run time, e.g.:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/build:/app/build" \
#       amylodb build --config configs/default.yaml
ENTRYPOINT ["amylodb"]
CMD ["--help"]
