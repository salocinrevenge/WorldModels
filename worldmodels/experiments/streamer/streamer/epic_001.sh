if [ ! -L "frames" ]; then  # link for dataset
    ln -s ../../datasets/epic_kitchens/frames/ frames
fi

exp_name='epic_001'
python -m train \
    --dataset "../../datasets/epic_kitchens/frames/P01"  \
    --sample "P01_01"\
    --name "$exp_name"_train \
    --p_device cpu \
    --p_n_gpus 1 \
    --dataset_split train \
    --dataset_percent 100 \
    \
    --max_layers 3 \
    --evolve_every 10000 \
    --buffer_size 20 \
    --force_fixed_buffer True \
    \
    --demarcation_mode average \
    --distance_mode similarity \
    \
    --lr 1e-4 \
    --alpha 3 \
    --optimize_every 100 \
    --average_every 100 \
    --optimize True \
    --save_every 25000 \
    \
    --dbg \
    --tb \
    --log_prefix /data/D2/datasets/ego4d/videos/ \
    --log_postfix MP4 \
    --log_base_every 1000 \


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi



python -m train \
    --dataset "../../datasets/epic_kitchens/frames/P01"  \
    --sample "P01_101"\
    --name "$exp_name"_test \
    --p_device cpu \
    --p_n_gpus 8 \
    --dataset_split test \
    --dataset_percent 100 \
    \
    --max_layers 3 \
    --init_ckpt $(ls -1 out/"$exp_name"_train/checkpoints/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -n 1) \
    --evolve_every 10000 \
    --buffer_size 20 \
    --force_fixed_buffer True \
    \
    --demarcation_mode average \
    --distance_mode similarity \
    \
    --optimize False \
    --dbg \
    --tb \
    --log_prefix /data/D2/datasets/ego4d/videos/ \
    --log_postfix MP4 \
    --log_base_every 1000  # isso chama a funcao  streamer.loggers.video_logger.model a cada 1000 iteracoes 


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Testing completed successfully"
else
    echo "Testing failed with exit code $?"
    exit $?
fi



python streamer/experiments/compare.py -i out/epic_001_train/jsons/0/ -gt "../../datasets/epic_kitchens/groundTruth/P01/"
python streamer/experiments/compare.py -i out/epic_001_test/jsons/0/ -gt "../../datasets/epic_kitchens/groundTruth/P01/"