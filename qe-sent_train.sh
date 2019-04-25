#!/usr/bin/env bash
export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export QE_DIR=qe-2017
export OUTPUT_DIR=QE2017_output
python run_regression.py \
  --task_name=qe-sent \
  --do_train=False \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$QE_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --do_lower_case=False \
  --output_dir=$OUTPUT_DIR
