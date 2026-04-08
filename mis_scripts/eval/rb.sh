for ((c=1000; c <= 32000; c = c * 2)); do
    python -u -m decoding.mis \
        --data ../datasets/rb_200_300_test.npz \
        --ckpt ckpts/mis_rb.ckpt \
        --sampling_steps 1 \
        --cycles $c --runs 1 \
        --keep_rate 0.5 \
        --batch_size 100 \
        --threads_over_batches 5
done

