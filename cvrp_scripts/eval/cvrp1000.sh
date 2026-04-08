CKPT_PATH=ckpts/cvrp1000.ckpt
DATA_PATH=../datasets/cvrp1000_capacity50_testset.npz


python -u -m decoding.cvrp \
    --capacity 50 \
    --data $DATA_PATH \
    --ckpt $CKPT_PATH \
    --keep_rate 0.6 \
    --two_opt_steps 40 \
    --gumbel_scale_factor 0. \
    --batch_size 16 \
    --runs 1 --cycles 5 \
    --sampling_steps 128 \
    --augment_level 1 \
    --threads_over_batches 4 \
    --heatmap_dtype uint8 --topk 10000


python -u -m decoding.cvrp \
    --capacity 50 \
    --data $DATA_PATH \
    --ckpt $CKPT_PATH \
    --keep_rate 0.6 \
    --two_opt_steps 40 \
    --gumbel_scale_factor 0. \
    --batch_size 16 \
    --runs 2 --cycles 5 \
    --sampling_steps 128 \
    --augment_level 1 \
    --threads_over_batches 4 \
    --heatmap_dtype uint8 --topk 10000


python -u -m decoding.cvrp \
    --capacity 50 \
    --data $DATA_PATH \
    --ckpt $CKPT_PATH \
    --keep_rate 0.6 \
    --two_opt_steps 40 \
    --gumbel_scale_factor 0. \
    --batch_size 16 \
    --runs 4 --cycles 5 \
    --sampling_steps 128 \
    --augment_level 1 \
    --threads_over_batches 4 \
    --heatmap_dtype uint8 --topk 10000


python -u -m decoding.cvrp \
    --capacity 50 \
    --data $DATA_PATH \
    --ckpt $CKPT_PATH \
    --keep_rate 0.6 \
    --two_opt_steps 40 \
    --gumbel_scale_factor 0. \
    --batch_size 16 \
    --runs 8 --cycles 5 \
    --sampling_steps 128 \
    --augment_level 1 \
    --threads_over_batches 4 \
    --heatmap_dtype uint8 --topk 10000


python -u -m decoding.cvrp \
    --capacity 50 \
    --data $DATA_PATH \
    --ckpt $CKPT_PATH \
    --keep_rate 0.6 \
    --two_opt_steps 40 \
    --gumbel_scale_factor 0. \
    --batch_size 16 \
    --runs 8 --cycles 10 \
    --sampling_steps 128 \
    --augment_level 1 \
    --threads_over_batches 4 \
    --heatmap_dtype uint8 --topk 10000


