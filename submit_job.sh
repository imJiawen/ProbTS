# multivariate datasets:
# ['exchange_rate_nips', 'solar_nips','electricity_nips', 'traffic_nips','wiki2000_nips']

# Univariate datasets:
# ['m4_weekly', 'm4_hourly', 'm4_daily', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly']

# Long-term forecasting:
# ['etth1', 'etth2','ettm1','ettm2','traffic_ltsf', 'electricity_ltsf', 'exchange_ltsf', 'illness_ltsf', 'weather_ltsf']


MODEL=timegrad
CTX_LEN=96

# revin=false
# scaling=true
# scaler=standard # identity, standard

for DATASET in 'etth1' 'exchange_ltsf' 'weather_ltsf'
do
    for PRED_LEN in 96 192 336 720
    do
        python run.py --config config/ltsf/${DATASET}/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${2} \
            --trainer.default_root_dir ${1}${5}_revin_${3}_scaling_${4} \
            --data.data_manager.init_args.split_val true \
            --trainer.max_epochs 40 \
            --data.data_manager.init_args.dataset ${DATASET} \
            --data.data_manager.init_args.context_length ${CTX_LEN} \
            --data.data_manager.init_args.prediction_length ${PRED_LEN} \
            --model.forecaster.init_args.use_scaling ${4} \
            --model.forecaster.init_args.revin ${3} \
            --data.batch_size 64 \
            --data.test_batch_size 64 \
            --trainer.limit_train_batches 100 \
            --trainer.accumulate_grad_batches 1 \
            --data.data_manager.init_args.scaler ${5}
    done
done