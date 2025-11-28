#!/bin/bash
#PBS -N memit_eval_gptj
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=40gb:ngpus=1:gpu_mem=26gb
#PBS -l walltime=0:30:00

source ./base_eval.sh

MODEL_NAME='EleutherAI/gpt-j-6B'
MODEL_PARAMS='EleutherAI_gpt-j-6B.json'
CHANGES_CNT=20 # how many facts to change
EDITS_AT_TIME=10 # how many facts are changed in a single run of MEMIT

# run memit
python3 -m experiments.evaluate \
	--alg_name=MEMIT \
	--model_name=$MODEL_NAME \
	--hparams_fname="$MODEL_PARAMS" \
	--num_edits=$EDITS_AT_TIME \
	--dataset_size_limit=$CHANGES_CNT \
	--save_deltas

if [ $? -ne 0 ]; then 
    echo >&2 "Error while running python MEMIT"; exit 3;
fi

# copy results
cp -r results $DATADIR

clean_scratch
