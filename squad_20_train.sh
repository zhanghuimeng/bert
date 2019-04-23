#!/usr/bin/env bash
export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export SQUAD_DIR=SQUAD
export OUTPUT_DIR=SQUAD_20_output
export OUTPUT_DIR_INIT=$OUTPUT_DIR/init
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --do_lower_case=False \
  --output_dir=$OUTPUT_DIR_INIT \
  --version_2_with_negative=True

# Run this script to tune a threshold for predicting null versus non-null answers
export THRESH = $(python $SQUAD_DIR/evaluate-v2.0.py \
    $SQUAD_DIR/dev-v2.0.json $OUTPUT_DIR_INIT/predictions.json \
    --na-prob-file $OUTPUT_DIR_INIT/null_odds.json)

# re-run the model to generate predictions with the derived threshold
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=256 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --do_lower_case=False \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
