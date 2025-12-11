#!/bin/bash
#PBS -N XXX
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=20gb:ngpus=1:gpu_mem=20gb
#PBS -l walltime=5:00:00

source /storage/brno2/home/xzvara01/memit/PBS/base_eval.sh

MODEL="gpt2"
MODEL_NAME='gpt2-xl'
MODEL_PARAMS='gpt2-xl.json'
DS='ct' # countertest dataset
CTNAME="countertest_birthplace"
ALG_NAME="ROME"

# run memit
python3 -m experiments.evaluate \
	--alg_name=${ALG_NAME}v2 \
	--model_name=$MODEL_NAME \
	--hparams_fname="$MODEL_PARAMS" \
	--ds_name=$DS \
	--ct_name=$CTNAME

if [ $? -ne 0 ]; then 
    echo >&2 "Error while running python MEMIT"; exit 3;
fi

# copy results
RESULTS="$DATADIR/results/${MODEL}_${ALG_NAME}_$CTNAME"
mkdir $RESULTS
cp -r results $RESULTS

clean_scratch
