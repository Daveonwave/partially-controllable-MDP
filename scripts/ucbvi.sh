python main.py \
    --env taxi \
    --algo ucbvi \
    --dest_folder "/data2/salaorni/pcmdp" \
    --exp_name "newTaxi" \
    --world "world.yaml" \
    --n_episodes 5000 \
    --gamma 1 \
    --tol 0.01 \
    --c_bonus 0.5 \
    --delta 0.000001 \
    --max_no_improvement 5000 \
    --eval_episodes 50 \
    --eval_every 20 \
    --train_seeds 1 \
    --eval_seed 1234

# python main.py \
#     --env elevator \
#     --algo ucbvi \
#     --dest_folder "/data2/salaorni/pcmdp" \
#     --exp_name "tinyElev" \
#     --world "tinyWorld.yaml" \
#     --n_episodes 5000 \
#     --gamma 1 \
#     --tol 0.01 \
#     --c_bonus 0.5 \
#     --delta 0.000001 \
#     --max_no_improvement 5000 \
#     --eval_episodes 50 \
#     --eval_every 20 \
#     --train_seeds 1 2 3 4 \
#     --eval_seed 1234

