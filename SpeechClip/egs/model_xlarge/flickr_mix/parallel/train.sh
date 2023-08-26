echo "[Train] SpeechCLIP Parallel Large on Mixed Flickr8k"
EXP_ROOT="exp_flickr_mix"
CFG="config/speechCLIP/model_xlarge/flickr_mix/spchclp_p.yaml"
mkdir $EXP_ROOT
python3 run_task.py \
    "TrainKWClip_GeneralTransformer" \
    --config $CFG \
    --gpus 1 \
    --njobs 4 \
    --seed 7122 \
    --train \
    --save_path $EXP_ROOT


