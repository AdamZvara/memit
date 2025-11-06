#!/bin/bash
#PBS -N memit_evaluate_eluther
#PBS -l select=1:ncpus=2:mem=64gb:scratch_local=30gb:ngpus=1:gpu_mem=26gb
#PBS -l walltime=0:30:00

export HF_HOME='/storage/brno2/home/xzvara01/HFCache'

export MODEL='gptxl'
export CHANGES_CNT=20 # how many facts to change
export EDITS_AT_TIME=10 # how many facts are changed in a single run of MEMIT

source configs/${MODEL}.sh

# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
DATADIR=/storage/brno2/home/xzvara01

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# test if the scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy memit into scratch directory
cp $DATADIR/memit  $SCRATCHDIR -r || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cd $SCRATCHDIR/memit 

# setup conda environment
if [ ! -d $HOME/.conda/ ]; then 
    ln -s "$DATADIR/.conda" $HOME/.conda 
fi
conda activate memit_original

# run memit
python3 -m experiments.evaluate \
	--alg_name=MEMIT \
	--model_name=$MODEL_NAME \
	--hparams_fname="$MODEL_PARAMS.json" \
	--num_edits=$EDITS_AT_TIME \
	--dataset_size_limit=$CHANGES_CNT \
	--save_deltas

if [ $? -ne 0 ]; then 
    echo >&2 "Error while running python MEMIT"; exit 3;
fi

# copy results
cp -r results $DATADIR

clean_scratch
