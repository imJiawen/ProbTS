export CUDA_VISIBLE_DEVICES=3

MODEL=patchtst

for dataset_name in bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H
do
    python run.py --config config/default/${MODEL}.yaml --seed_everything 0  \
                --seed_everything 0 \
                --data.data_manager.init_args.dataset gift/${dataset_name}/short \
                --data.data_manager.init_args.path /data/Blob_WestJP/v-jiawezhang/datasets \
                --trainer.default_root_dir /data/Blob_WestJP/v-jiawezhang/log/gift_eval \
                --trainer.max_epochs 50 \
                --data.test_batch_size 1
done