python main.py \
    --env taxi \
    --algo exa_ql \
    --exp_name "prova" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "world.yaml" \
    --n_episodes 2000 \
    --gamma 1 \
    --epsilon 1.0 \
    --epsilon_decay 0.99 \
    --epsilon_min 0.05 \
    --alpha 0.1 \
    --tol 0.01 \
    --max_no_improvement 2000 \
    --eval_episodes 50 \
    --eval_every 1 \
    --train_seed 1 2 3 \
    --eval_seed 1234 

# python main.py \
#     --env elevator \
#     --algo exa_ql \
#     --exp_name "prova" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "world.yaml" \
#     --n_episodes 20000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.995 \
#     --epsilon_min 0.05 \
#     --alpha 0.01 \
#     --tol 0.01 \
#     --max_no_improvement 20000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1  \
#     --eval_seed 1234 


# python main.py \
#     --env elevator \
#     --algo exa_ql \
#     --exp_name "tinyElev" \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 1000 \
#     --gamma 1 \
#     --epsilon 1.0 \
#     --epsilon_decay 0.995 \
#     --epsilon_min 0.05 \
#     --alpha 0.01 \
#     --tol 0.01 \
#     --max_no_improvement 20000 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 \
#     --eval_seed 1234 
