#!/usr/bin/env bash
export SQUAD_DIR=SQUAD
export OUTPUT_DIR=SQUAD_11_output
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUTPUT_DIR/predictions.json
