set -a

COUNTERFACTS_SUBS=("counterfact_sub_instrument" "counterfact_sub_profession" "counterfact_sub_country" "counterfact_sub_birthplace" "counterfact_sub_sport")
COUNTERTESTS=("countertest_instrument" "countertest_profession" "countertest_country" "countertest_birthplace")

for i in ${COUNTERTESTS[@]}; do
	sed -i "s/^CTNAME=.*/CTNAME=\"$i\"/" eval_gpt2.sh
	sed -i "s/^#PBS -N.*/#PBS -N gpt2_$i/" eval_gpt2.sh 
	qsub eval_gptj.sh
done

