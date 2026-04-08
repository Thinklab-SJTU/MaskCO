python -u -m training.train_cvrp \
    --num_nodes 1000 \
    --data_augment 1 \
    --capacity 50 \
    --model_config qk_norm_base_dim512 \
    --peak_lr 8e-3 \
    --batch_size 512 \
    --num_steps 150000 \
    --num_warmup_steps 50 \
    --save_interval 2500 \
    --data ../datasets/cvrp1000_train_merged.npz \
    --logdir logs/cvrp1000_base_dim512_merge_wd1e-1 \
    --savedir ckpts/cvrp1000_base_dim512_merge_wd1e-1 \
    --optimizer_type muon_decay --weight_decay 1e-1 \

