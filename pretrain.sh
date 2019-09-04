#!/usr/bin/env bash

trainAndEval() {
    type=$1
    input_file=$2
    parsed_file=$3
    output_dir=$4
    steps=$5
    model=$6
    tune_output_dir=$7

    BERT_BASE_DIR=/data/disk3/private/zhm/201909_BERT/dep/model/multi_cased_L-12_H-768_A-12
    GLUE_DIR=/data/disk3/private/zhm/201909_BERT/dep/data/data/

#    rm -f $parsed_file
#    rm -rf $output_dir
    rm -rf $tune_output_dir

    # make data
    python create_pretraining_data.py \
      --input_file=$input_file \
      --output_file=$parsed_file \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --do_lower_case=True \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5 \
      --mask_pos_type=$type

    # run pretraining
    python run_pretraining.py \
      --input_file=$parsed_file \
      --output_dir=$output_dir \
      --do_train=True \
      --do_eval=True \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --train_batch_size=24 \
      --max_seq_length=128 \
      --max_predictions_per_seq=20 \
      --num_train_steps=$steps \
      --num_warmup_steps=10 \
      --learning_rate=2e-5

    # run finetuning
    python run_classifier.py \
      --task_name=MRPC \
      --do_train=true \
      --do_eval=true \
      --data_dir=$GLUE_DIR/MRPC \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$model \
      --max_seq_length=128 \
      --train_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=$tune_output_dir
}

TYPE=$1
STEPS=$2

BERT_BASE_DIR=/data/disk3/private/zhm/201909_BERT/dep/model/multi_cased_L-12_H-768_A-12
INPUT_FILE=./data/300k.en
PARSED_FILE=./tmp/$TYPE/tf_examples.tfrecord
OUTPUT_DIR=./tmp/pretraining_output/$TYPE
MODEL="./tmp/pretraining_output/$TYPE/model.ckpt-$STEPS"
TUNE_OUTPUT_DIR=./tmp/mrpc_output/$TYPE

mkdir ./tmp/$TYPE
trainAndEval $TYPE $INPUT_FILE $PARSED_FILE $OUTPUT_DIR $STEPS $MODEL $TUNE_OUTPUT_DIR
