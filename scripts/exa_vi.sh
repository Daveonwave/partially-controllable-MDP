python main.py \
    --env taxi \
    --algo exa_vi \
    --exp_name "SpawnProb0_6" \
    --dest_folder "/data2/salaorni/pcmdp" \
    --world "world.yaml" \
    --n_episodes 100 \
    --gamma 1 \
    --tol 0.01 \
    --max_no_improvement 100 \
    --eval_episodes 50 \
    --eval_every 1 \
    --train_seeds 1 2 3 4 \
    --eval_seed 1234


# python main.py \
#     --env elevator \
#     --algo exa_vi \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --exp_name "tinyElev" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 100 \
#     --gamma 1 \
#     --tol 0.01 \
#     --max_no_improvement 100 \
#     --eval_episodes 50 \
#     --eval_every 1 \
#     --train_seeds 1 2 3\
#     --eval_seed 1234
