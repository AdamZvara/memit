set -a

# Usage
# ./run_diversity.sh [gpt2|gptj] [ROME|MEMIT]
usage() {
	echo "Usage: $0 [gpt2|gptj] [ROME|MEMIT]"
	exit 1
}

# Check if GPT-2 or GPT-J
if [ "$1" == "gpt2" ]; then
	FILE=eval_gpt2.sh
	MODEL="gpt2"
elif [ "$1" == "gptj" ]; then
	FILE=eval_gptj.sh
	MODEL="gptj"
else
	usage
fi

# Check if ROME or MEMIT
if [ "$2" == "ROME" ]; then
	ALG_NAME="ROME"
elif [ "$2" == "MEMIT" ]; then
	ALG_NAME="MEMIT"
else
	usage
fi

COUNTERFACTS_SUBS=("counterfact_sub_instrument" "counterfact_sub_profession" "counterfact_sub_country" "counterfact_sub_birthplace" "counterfact_sub_sport")
COUNTERTESTS=("countertest_instrument" "countertest_profession" "countertest_country" "countertest_birthplace")

# Pick manually which countertests to run
for i in ${COUNTERTESTS[@]}; do
	sed -i "s/^#PBS -N.*/#PBS -N ${MODEL}_${ALG_NAME}_$i/" $FILE 
	sed -i "s/^CTNAME=.*/CTNAME=\"$i\"/" $FILE
	sed -i "s/^ALG_NAME=.*/ALG_NAME=\"$ALG_NAME\"/" $FILE
	qsub $FILE
done

