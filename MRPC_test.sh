#!/usr/bin/env bash
export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export GLUE_DIR=glue_data
export OUTPUT_DIR=MRPC_output
export TRAINED_CLASSIFIER=MRPC_output

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --do_lower_case=False \
  --output_dir=$OUTPUT_DIR
