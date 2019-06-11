#!/bin/sh

rm -rf __pycache__/
#
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 15 --p 0.1
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 25 --p 0.1
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 50 --p 0.1
#
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 15 --p 0.25
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 25 --p 0.25
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 50 --p 0.25
#
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 15 --p 0.5
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 25 --p 0.5
python test.py --model bayesian --snapshot ../out/snapshots/bayesian_best.pth.tar --mc 50 --p 0.5
