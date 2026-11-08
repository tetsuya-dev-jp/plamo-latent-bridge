#!/usr/bin/env bash
set -euo pipefail

PYTHON=.venv/bin/python

$PYTHON -m plamo_bridge data --mgsm
$PYTHON -m plamo_bridge features --splits train,val,test,heldout,mgsm
$PYTHON -m plamo_bridge translate --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-text --conditions text_ja,text_en,text_plamo_en --splits test,heldout
$PYTHON -m plamo_bridge evaluate-text --conditions text_ja,text_plamo_en --splits mgsm

for seed in 11 23 37; do
  $PYTHON -m plamo_bridge train --condition bridge --layer 8 --prefix-length 16 --seed "$seed"
done

$PYTHON -m plamo_bridge train --condition bridge --layer 4 --prefix-length 16 --seed 11
$PYTHON -m plamo_bridge train --condition bridge --layer 16 --prefix-length 16 --seed 11
$PYTHON -m plamo_bridge train --condition bridge --layer 8 --prefix-length 4 --seed 11
$PYTHON -m plamo_bridge train --condition soft_prefix --prefix-length 16 --seed 11

for seed in 11 23 37; do
  $PYTHON -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed "$seed" --splits test,heldout,mgsm
done

$PYTHON -m plamo_bridge evaluate-bridge --layer 4 --prefix-length 16 --seed 11 --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-bridge --layer 16 --prefix-length 16 --seed 11 --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 4 --seed 11 --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 11 --untrained --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-bridge --layer 8 --prefix-length 16 --seed 11 --shuffle --splits test,heldout,mgsm
$PYTHON -m plamo_bridge evaluate-bridge --condition soft_prefix --prefix-length 16 --seed 11 --splits test,heldout,mgsm

$PYTHON -m plamo_bridge analyze
