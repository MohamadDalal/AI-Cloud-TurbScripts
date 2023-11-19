#!/bin/sh

srun --gres=gpu:1 --time=1-00 singularity exec --nv ~/pytorch_23.09-py3.sif python main.py > log.txt
