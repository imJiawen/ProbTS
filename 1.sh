# export CUDA_VISIBLE_DEVICES=0

# short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

# med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"


# MODEL=patchtst

# for dataset_name in covid_deaths M_DENSE/H M_DENSE/D us_births/D us_births/M saugeenday/D saugeenday/W 
# do
#     python run.py --config config/default/${MODEL}.yaml --seed_everything 0  \
#                 --seed_everything 0 \
#                 --data.data_manager.init_args.dataset gift/${dataset_name}/short \
#                 --data.data_manager.init_args.path /data/Blob_WestJP/v-jiawezhang/datasets \
#                 --trainer.default_root_dir /data/Blob_WestJP/v-jiawezhang/log/gift_eval \
#                 --trainer.max_epochs 50 \
#                 --data.test_batch_size 1
# done


dataset_name=bizitobs_application

python run.py --config config/default/mean.yaml \
              --seed_everything 0 \
              --model.forecaster.init_args.mode batch \
              --data.data_manager.init_args.dataset gift/${dataset_name}/short \
              --data.data_manager.init_args.path /data/Blob_WestJP/v-jiawezhang/datasets \
              --trainer.default_root_dir ./exps