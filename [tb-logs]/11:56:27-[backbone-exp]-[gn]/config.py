act_type = 'relu'
cifar_data_root = 'DATA'
ckp_config = dict(
    dirpath='06-01-24/11:56:27-[backbone-exp]-[gn]',
    every_n_epochs=None,
    save_last=None)
ckp_root = '06-01-24/11:56:27-[backbone-exp]-[gn]'
device = 'cuda'
enable_wandb = True
load_weight_from = None
lr_sche_config = dict(config=dict(), type='constant')
model_config = dict(
    act_type='relu',
    base_block_config=dict(
        act='relu',
        in_channels=256,
        norm_config=dict(
            config=dict(num_channels=256, num_groups=16), type='gn'),
        out_channels=256),
    block_type='TwoConvBlock',
    channels=[
        32,
        64,
        128,
        256,
        256,
    ],
    norm_type='gn',
    num_block_per_stage=1,
    num_channels_per_gn_group=16)
norm_type = 'gn'
num_ep = 100
optimizer_config = dict(config=dict(lr=0.0003), type='adamw')
resume_ckpt_path = None
run_name = 'gn'
test_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
train_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
trainer_config = dict(
    check_val_every_n_epoch=1, log_every_n_steps=5, precision='32')
wandb_config = dict(offline=True, project='backbone-exp')
