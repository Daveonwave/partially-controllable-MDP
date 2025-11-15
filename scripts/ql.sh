# python main.py \
#     --env taxi \
#     --algo ql \
#     --exp_name "10seeds" \
#     --world "world.yaml" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --n_episodes 2000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.999 \
#     --epsilon_min 0.05 \
#     --alpha 0.1 \
#     --tol 0.01 \
#     --max_no_improvement 2000 \
#     --eval_episodes 50 \
#     --eval_every 1  \
#     --train_seeds 1 2 3 4 5 6 7 8 9 10 \
#     --eval_seed 1234

# python main.py \
#     --env elevator \
#     --algo ql \
#     --exp_name "10seeds" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 30000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.9995 \
#     --epsilon_min 0.05 \
#     --alpha 0.1 \
#     --tol 0.01 \
#     --max_no_improvement 10000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 2 3 4 5 6 7 8 9 10 \
#     --eval_seed 1234


python main.py \
    --env elevator \
    --algo ql \
    --exp_name "tinyElev" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "tinyWorld.yaml" \
    --n_episodes 30000 \
    --gamma 1 \
    --epsilon 1.0 \
    --epsilon_decay 0.9995 \
    --epsilon_min 0.05 \
    --alpha 0.1 \
    --tol 0.01 \
    --max_no_improvement 10000 \
    --eval_episodes 50 \
    --eval_every 5 \
    --train_seeds 1 2 3 4 \
    --eval_seed 1234
