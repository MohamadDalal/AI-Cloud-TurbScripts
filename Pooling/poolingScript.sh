#!/bin/sh

srun --time=1-00 singularity exec ~/pytorch_23.09-py3.sif python 3dpooling_blur.py
