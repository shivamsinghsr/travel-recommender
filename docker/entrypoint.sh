#!/bin/sh
# One image, three roles:  api (default) | train (one run) | scheduler (retrain every TRAIN_INTERVAL_SECONDS)
set -e

case "${1:-api}" in
  api)
    python -m api.seed
    if [ "${TRAIN_ON_START:-0}" = "1" ]; then
      python -m api.train --if-missing || echo "training failed; serving the live-fit model until the next run"
    fi
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --workers "${WEB_CONCURRENCY:-2}" --proxy-headers --forwarded-allow-ips="*"
    ;;
  train)
    exec python -m api.train
    ;;
  scheduler)
    while true; do
      sleep "${TRAIN_INTERVAL_SECONDS:-86400}"
      python -m api.train || echo "training run failed; keeping the active model"
    done
    ;;
  *)
    exec "$@"
    ;;
esac
