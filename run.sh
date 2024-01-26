mkdir logs/$2 saved_models/$2 results/$2

python train.py $1
python eval.py $1
mv logs/$1* logs/$2/.
mv saved_models/ppo_$1* saved_models/$2/.
mv results/ppo_$1* results/$2/.