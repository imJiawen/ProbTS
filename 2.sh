export CUDA_VISIBLE_DEVICES=1

MODEL=elastst

for dataset_name in us_births/D us_births/M saugeenday/D saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D
do
    python run.py --config config/multi_hor/${MODEL}.yaml --seed_everything 0  \
                --seed_everything 0 \
                --data.data_manager.init_args.dataset gift/${dataset_name}/short \
                --data.data_manager.init_args.path /data/Blob_WestJP/v-jiawezhang/datasets \
                --trainer.default_root_dir /data/Blob_WestJP/v-jiawezhang/log/gift_eval \
                --trainer.max_epochs 50 \
                --data.test_batch_size 1
done