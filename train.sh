#!/bin/sh

rm -rf __pycache__/
BS=128
python -u train.py --bs ${BS} --epoch 10 --model baseline | tee `date '+%Y-%m-%d_%H-%M-%S'`_baseline.log
python -u train.py --bs ${BS} --epoch 10 --model bayesian | tee `date '+%Y-%m-%d_%H-%M-%S'`_bayesian.log
