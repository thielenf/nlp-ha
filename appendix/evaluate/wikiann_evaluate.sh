#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/home/thielen/nlp-ha
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/media/data/thielen/nlp-ha/outputs/github_mrc/wikiann/large_lr2e-5_drop0.2_norm1.0_bsz32_hard_span_weight0.1_warmup0_maxlen180
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=5_v4.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=4

python3 ${REPO_PATH}/evaluate/mrc_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID}