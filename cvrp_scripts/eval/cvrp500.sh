CKPT_PATH=ckpts/cvrp500.ckpt

for ((c=5; c <= 80; c = c * 2)); do
    python -u -m decoding.cvrp \
        --capacity 50 \
        --data ../datasets/cvrp500_capacity50_testset128_seed9.npz \
        --ckpt $CKPT_PATH \
        --keep_rate 0.3 \
        --two_opt_steps 20 \
        --gumbel_scale_factor 0. \
        --batch_size 32 \
        --runs 8 --cycles $c \
        --sampling_steps 16 \
        --augment_level 1 \
        --threads_over_batches 4 \
        --heatmap_dtype uint8 --topk 10000
done

