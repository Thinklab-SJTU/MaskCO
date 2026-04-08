CKPT_PATH=ckpts/cvrp100.ckpt

for ((c=40; c <= 640; c = c * 2)); do
    python -u -m decoding.cvrp \
        --capacity 50 \
        --data ../datasets/cvrp100_testset1280_seed88_subset.npz \
        --ckpt $CKPT_PATH \
        --keep_rate 0.3 \
        --two_opt_steps 4 \
        --gumbel_scale_factor 0. \
        --batch_size 128 \
        --runs 8 --cycles $c \
        --sampling_steps 2 \
        --augment_level 1 \
        --threads_over_batches 5 \
        # 2>/dev/null
done

