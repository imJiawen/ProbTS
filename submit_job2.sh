# multivariate datasets:
# ['exchange_rate_nips', 'solar_nips','electricity_nips', 'traffic_nips','wiki2000_nips']

# Univariate datasets:
# ['m4_weekly', 'm4_hourly', 'm4_daily', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly']

# Long-term forecasting:
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']


# MODEL=csdi
CTX_LEN=96


scaler=standard # identity, standard
# DATASET='illness_ltsf'


revin=true
scaling=false

for DATASET in 'etth2' 'ettm1' 'ettm2'
do
    python run.py --config config/ltsf/${DATASET}/${4}.yaml --seed_everything 0  \
        --data.data_manager.init_args.path ${2} \
        --trainer.default_root_dir ${1}${scaler}_revin_${revin}_scaling_${scaling} \
        --data.data_manager.init_args.split_val true \
        --trainer.max_epochs 40 \
        --data.data_manager.init_args.dataset ${DATASET} \
        --data.data_manager.init_args.context_length ${CTX_LEN} \
        --data.data_manager.init_args.prediction_length ${3} \
        --model.forecaster.init_args.use_scaling ${scaling} \
        --model.forecaster.init_args.revin ${revin} \
        --data.batch_size 16 \
        --data.test_batch_size 16 \
        --trainer.limit_train_batches 200 \
        --trainer.accumulate_grad_batches 2 \
        --data.data_manager.init_args.scaler $scaler
done


# revin=false
# scaling=false

# for DATASET in 'etth2' 'ettm1' 'ettm2'
# do
#     python run.py --config config/ltsf/${DATASET}/${4}.yaml --seed_everything 0  \
#         --data.data_manager.init_args.path ${2} \
#         --trainer.default_root_dir ${1}${scaler}_revin_${revin}_scaling_${scaling} \
#         --data.data_manager.init_args.split_val true \
#         --trainer.max_epochs 40 \
#         --data.data_manager.init_args.dataset ${DATASET} \
#         --data.data_manager.init_args.context_length ${CTX_LEN} \
#         --data.data_manager.init_args.prediction_length ${3} \
#         --model.forecaster.init_args.use_scaling ${scaling} \
#         --model.forecaster.init_args.revin ${revin} \
#         --data.batch_size 16 \
#         --data.test_batch_size 16 \
#         --trainer.limit_train_batches 200 \
#         --trainer.accumulate_grad_batches 2 \
#         --data.data_manager.init_args.scaler $scaler
# done

# revin=false
# scaling=true

# for DATASET in 'etth2' 'ettm1' 'ettm2'
# do
#     python run.py --config config/ltsf/${DATASET}/${4}.yaml --seed_everything 0  \
#         --data.data_manager.init_args.path ${2} \
#         --trainer.default_root_dir ${1}${scaler}_revin_${revin}_scaling_${scaling} \
#         --data.data_manager.init_args.split_val true \
#         --trainer.max_epochs 40 \
#         --data.data_manager.init_args.dataset ${DATASET} \
#         --data.data_manager.init_args.context_length ${CTX_LEN} \
#         --data.data_manager.init_args.prediction_length ${3} \
#         --model.forecaster.init_args.use_scaling ${scaling} \
#         --model.forecaster.init_args.revin ${revin} \
#         --data.batch_size 16 \
#         --data.test_batch_size 16 \
#         --trainer.limit_train_batches 200 \
#         --trainer.accumulate_grad_batches 2 \
#         --data.data_manager.init_args.scaler $scaler
# done