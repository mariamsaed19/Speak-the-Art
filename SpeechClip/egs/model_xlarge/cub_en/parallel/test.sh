echo "[Test] SpeechCLIP Parallel Large on CUB"
EXP_ROOT="exp_cub_en"
DATASET_ROOT="data/CUB_200_2011"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "slt_ckpts/SpeechCLIP/large/cub/parallel/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt" \
    --dataset_root $DATASET_ROOT \
    --gpus 1 \
    --njobs 4 \
    --seed 7122 \
    --test \
    --save_path $EXP_ROOT


