#!/bin/bash
#PBS -N memit_evaluate
#PBS -l select=1:ncpus=2:mem=32gb:scratch_local=10gb:ngpus=1:gpu_mem=16gb
#PBS -l walltime=1:00:00 

# define a DATADIR variable: directory where the input files are taken from and where the output will be copied to
DATADIR=/storage/plzen1/home/xzvara01

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
    ln -s /storage/plzen1/home/xzvara01/.conda $HOME/.conda 
fi
conda activate memit_original

# run memit
EDITS=1000
python3 -m experiments.evaluate --alg_name=MEMIT --model_name=gpt2-xl --hparams_fname=gpt2-xl.json --num_edits=$EDITS
if [ $? -ne 0 ]; then 
    echo >&2 "Error while running python MEMIT"; exit 3;
fi

# copy results
cp -r results $DATADIR/results

clean_scratch
