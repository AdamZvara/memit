#!/bin/bash
#PBS -N gptj_counterfact_sub_country
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=64gb:ngpus=1:gpu_mem=35gb
#PBS -l walltime=5:30:00

source /storage/brno2/home/xzvara01/memit/PBS/base_eval.sh

MODEL_NAME='EleutherAI/gpt-j-6B'
MODEL_PARAMS='EleutherAI_gpt-j-6B.json'
DS='ct'
CTNAME="counterfact_sub_country"

# run memit
python3 -m experiments.evaluate \
	--alg_name=ROMEv2 \
	--model_name=$MODEL_NAME \
	--hparams_fname="$MODEL_PARAMS" \
	--ds_name=$DS \
	--ct_name=$CTNAME

if [ $? -ne 0 ]; then 
    echo >&2 "Error while running python MEMIT"; exit 3;
fi

# copy results
RESULTS="$DATADIR/results/gptj/$CTNAME"
mkdir $RESULTS
cp -r results $RESULTS

clean_scratch
