export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python -u -m training.train_mis \
    --model_config dim256_qknorm_sigmoid_decoder_only_layer24 \
    --peak_lr 1e-3 \
    --batch_size 256 \
    --num_steps 192000 \
    --num_warmup_steps 50 \
    --save_interval 2500 \
    --data ../datasets/er_700_800_train_repaired.npz --label ../datasets/er_700_800_train_repaired_labels.npz \
    --logdir logs/mis_er_l24 \
    --savedir ckpts/mis_er_l24 \
    --optimizer_type muon 
