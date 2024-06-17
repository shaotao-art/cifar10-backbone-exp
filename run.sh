device_id=7
config_p=./configs/config.py
run_name=tmp

CUDA_VISIBLE_DEVICES=$device_id  python run.py \
                    --config $config_p \
                    --run_name $run_name