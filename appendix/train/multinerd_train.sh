#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: genia.sh

REPO_PATH=/home/thielen/nlp-ha
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/media/data/thielen/nlp-ha/datasets/MultiNERD/de/mrc-ner
BERT_DIR=/media/data/models/gbert-base

BATCH=8
GRAD_ACC=4
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
LR_MINI=3e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=180
MAX_NORM=1.0
MAX_EPOCH=6
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold
OUTPUT_DIR=/media/data/thielen/nlp-ha/outputs/github_mrc/multinerd/large_lr${LR}_drop${MRC_DROPOUT}_norm${MAX_NORM}_bsz32_hard_span_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAX_LEN}
mkdir -p ${OUTPUT_DIR}

source .venv/bin/activate
#pip install -r versions.txt

# --distributed_backend=dp \
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=5 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--batch_size ${BATCH} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--lr_mini ${LR_MINI} \
--flat \
--workers 8
