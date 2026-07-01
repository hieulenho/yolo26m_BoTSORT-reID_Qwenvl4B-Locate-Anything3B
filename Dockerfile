FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FOOTBALL_TRACKING_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements/base.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY demo ./demo

RUN python -m pip install --editable .

CMD ["python", "-m", "football_tracking.cli", "doctor"]
