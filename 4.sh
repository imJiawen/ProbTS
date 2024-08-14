export CUDA_VISIBLE_DEVICES=0

DATA_DIR='/data/Blob_WestJP/v-jiawezhang/data/all_datasets/'
LOG_DIR=/data/Blob_WestJP/v-jiawezhang/log/abl_norm/

# multivariate datasets:
# ['exchange_rate_nips', 'solar_nips','electricity_nips', 'traffic_nips','wiki2000_nips']

# Univariate datasets:
# ['m4_weekly', 'm4_hourly', 'm4_daily', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly']

# Long-term forecasting:
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']
# NOTE: when using long-term forecasting datasets, please explicit assign context_length and prediction_length, e.g., :
# --data.data_manager.init_args.context_length 96 \
# --data.data_manager.init_args.prediction_length 192 \

# run pipeline with train and test
# replace ${MODEL} with tarfet model name, e.g, patchtst
# replace ${DATASET} with dataset name

# if not specify dataset_path, the default path is ./datasets

MODEL=csdi
CTX_LEN=96

# scaler=identity # identity, standard


revin=true
scaling=false

for DATASET in 'wiki2000_nips' 'traffic_nips' 'electricity_nips' 'solar_nips' 'exchange_rate_nips' 
do
    for scaler in standard
    do
        python run.py --config config/default/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR}${scaler}_revin_${revin}_scaling_${scaling} \
            --data.data_manager.init_args.split_val true \
            --trainer.max_epochs 40 \
            --data.data_manager.init_args.dataset ${DATASET} \
            --model.forecaster.init_args.use_scaling ${scaling} \
            --model.forecaster.init_args.revin ${revin} \
            --data.batch_size 64 \
            --data.test_batch_size 64 \
            --trainer.limit_train_batches 100 \
            --trainer.accumulate_grad_batches 1 \
            --data.data_manager.init_args.scaler ${scaler}
    done
done