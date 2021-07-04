#! /bin/bash

eval "$(/home/jonathan/anaconda3/condabin/conda shell.bash hook)"
conda activate ./env
python ./call_api.py
