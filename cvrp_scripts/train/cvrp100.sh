python -u -m training.train_cvrp \
    --num_nodes 100 \
    --capacity 50 \
    --data_augment 3 \
    --model_config qk_norm_base_dim512 \
    --peak_lr 1e-3 \
    --batch_size 8192 \
    --num_steps 225000 \
    --num_warmup_steps 50 \
    --save_interval 5000 \
    --data ../datasets/cvrp100_1536000_seed1234+4321+5678.npz \
    --logdir logs/cvrp100_base_dim512_wd2e-1 \
    --savedir ckpts/cvrp100_base_dim512_wd2e-1 \
    --optimizer_type muon_decay --weight_decay 2e-1 \

