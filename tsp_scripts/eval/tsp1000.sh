CKPT_PATH=ckpts/tsp1000.ckpt
for ((c=20; c <= 160; c = c * 2)); do
    python -u -m decoding.tsp \
        --data ../datasets/tsp1000_concorde_23.11812.npz \
        --ckpt $CKPT_PATH \
        --batch_size 16 \
        --sampling_steps 2 \
        --two_opt_steps 10 \
        --cycles $c --runs 8 \
        --keep_rate 0.1 \
        --threads_over_batches 2 \
        --heatmap_dtype uint8 --topk 20000 \
        --augment_level 1 \
        # 2> /dev/null
done
