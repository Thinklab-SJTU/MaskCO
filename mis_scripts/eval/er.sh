for ((c=1000; c <= 32000; c = c * 2)); do
    python -u -m decoding.mis \
        --data ../datasets/er_700_800_test.npz \
        --ckpt ckpts/mis_er.ckpt \
        --sampling_steps 1 \
        --cycles $c --runs 1 \
        --keep_rate 0.6 \
        --batch_size 32 \
        --threads_over_batches 4
done
