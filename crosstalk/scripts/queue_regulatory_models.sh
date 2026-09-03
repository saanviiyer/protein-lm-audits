#!/bin/bash
# Serial queue for the regulatory model arm. One job at a time: the host is
# shared and MPS plus sklearn already saturate it.
cd "$(dirname "$0")/.."
export HF_HOME=/Users/saanviiyer/Downloads/CALTECH/RESEARCH/AITHYRA/.hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=./.venv-glm/bin/python

# 1. Paired model-minus-trivial bootstrap on identical test rows.
$PY scripts/run_regulatory_headtohead.py --model nt    > results/regulatory_models_headtohead_nt.log 2>&1
$PY scripts/run_regulatory_headtohead.py --model hyena > results/regulatory_models_headtohead_hyena.log 2>&1

# 2. Positional readout, on the two tasks a positional one-hot wins. Mean pooling
#    is position-blind, so without this the splice-site gap could be blamed on the
#    readout rather than the representation. win8 keeps coarse position at a
#    feature count of the same order as the one-hot's 4L.
for M in nt hyena; do
  D=L12_win8; [ $M = hyena ] && D=L4_win8
  $PY scripts/run_regulatory_models.py --model $M --poolings win8 --default-config $D \
      --tasks splice_sites_all promoter_tata \
      --out results/regulatory_models_positional_$M.csv > results/regulatory_models_positional_$M.log 2>&1
done
echo QUEUE_DONE
