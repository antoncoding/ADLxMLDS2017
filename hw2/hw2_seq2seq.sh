#!/bin/bash
wget https://www.dropbox.com/s/lj8ce9xg7dr5w9f/v728_model_2300_256.h5
python model_seq2seq_run.py $1 $2 $3
