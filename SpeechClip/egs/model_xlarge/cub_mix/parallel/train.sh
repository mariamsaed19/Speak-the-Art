echo "[Train] SpeechCLIP Parallel Large on CUB"
EXP_ROOT="exp_cub_mix"
CFG="config/speechCLIP/model_xlarge/cub_mix/spchclp_p.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "exp_cub_mix/last.ckpt" \
    --config $CFG \
    --gpus 1 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


