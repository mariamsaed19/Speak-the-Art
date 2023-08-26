echo "[Train] SpeechCLIP Parallel Large on Flickr8k"
EXP_ROOT="exp_flickr_en"
CFG="config/speechCLIP/model_xlarge/flickr_en/spchclp_p.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --resume "exp_flickr_en/last.ckpt" \
    --config $CFG \
    --gpus 1 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


