export CUDA_VISIBLE_DEVICES=1
MODEL=patchtst
DATASET=etth1
CTX_LEN=96
PRED_LEN=96

DATA_DIR='/data/Blob_WestJP/v-jiawezhang/data/all_datasets/'
LOG_DIR=/data/Blob_WestJP/v-jiawezhang/log/nn_baseline_scale/

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

MODEL=gru
CTX_LEN=96
for DATASET in 'etth1' 'etth2' 'ettm1' 'ettm2' 
do
    for PRED_LEN in 96 192 336 720
    do
        python run.py --config config/default/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR} \
            --data.data_manager.init_args.split_val true \
            --trainer.max_epochs 50 \
            --data.data_manager.init_args.dataset ${DATASET} \
            --data.data_manager.init_args.context_length ${CTX_LEN} \
            --data.data_manager.init_args.prediction_length ${PRED_LEN} \
            --model.forecaster.init_args.use_scaling true
    done
done