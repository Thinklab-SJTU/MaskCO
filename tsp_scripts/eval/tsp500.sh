CKPT_PATH=ckpts/tsp500.ckpt
for ((c=40; c <= 320; c = c * 2)); do
    python -u -m decoding.tsp \
        --data ../datasets/tsp500_concorde_16.54581.npz \
        --ckpt $CKPT_PATH \
        --batch_size 16 \
        --sampling_steps 1 \
        --two_opt_steps 5 \
        --cycles $c --runs 8 \
        --keep_rate 0.2 \
        --threads_over_batches 2 \
        --heatmap_dtype uint8 --topk 5000 \
        --augment_level 1 \
        # 2> /dev/null
done
