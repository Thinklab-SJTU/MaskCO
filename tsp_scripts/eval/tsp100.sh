CKPT_PATH=ckpts/tsp100.ckpt
for ((c=40; c <= 320; c = c * 2)); do
    python -u -m decoding.tsp \
        --data ../datasets/tsp100_concorde_7.75585.npz \
        --ckpt $CKPT_PATH \
        --batch_size 128 \
        --sampling_steps 1 \
        --two_opt_steps 1 \
        --cycles $c --runs 8 \
        --keep_rate 0.2 \
        --threads_over_batches 2 \
        --augment_level 1 \
        # 2> /dev/null
done
