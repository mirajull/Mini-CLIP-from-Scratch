#!/usr/bin/env bash
set -e

CONFIG_PATH="configs/default.yaml"

python -m src.train --config "${CONFIG_PATH}"
