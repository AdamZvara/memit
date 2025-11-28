#!/bin/bash
#PBS -N memit_eval_gpt2
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=20gb:ngpus=1:gpu_mem=10gb
#PBS -l walltime=0:20:00

source /storage/brno2/home/xzvara01/memit/PBS/base_eval.sh

MODEL_NAME='gpt2-xl'
MODEL_PARAMS='gpt2-xl.json'
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
