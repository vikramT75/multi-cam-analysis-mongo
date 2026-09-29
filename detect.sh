#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No configuration file provided."
    echo "Usage: ./detect.sh <config.yaml>"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

VENV_PYTHON="venv/Scripts/python.exe"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python"
fi

echo "Running detector with config: $CONFIG_FILE"
$VENV_PYTHON src/detector.py --config "$CONFIG_FILE"
