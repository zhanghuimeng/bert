#!/usr/bin/env bash
export SQUAD_DIR=SQUAD
export OUTPUT_DIR=SQUAD_20_output
python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json $OUTPUT_DIR/predictions.json
