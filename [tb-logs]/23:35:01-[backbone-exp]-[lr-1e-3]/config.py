act_type = 'relu'
cifar_data_root = 'DATA'
ckp_config = dict(
    dirpath='05-31-24/23:35:01-[backbone-exp]-[lr-1e-3]',
    every_n_epochs=None,
    save_last=True)
ckp_root = '05-31-24/23:35:01-[backbone-exp]-[lr-1e-3]'
device = 'cuda'
enable_wandb = True
load_weight_from = None
lr_sche_config = dict(config=dict(), type='constant')
model_config = dict(
    act_type='relu',
    base_block_config=dict(
        act='relu',
        in_channels=512,
        norm_config=dict(config=dict(num_features=512), type='bn'),
        out_channels=512),
    block_type='TwoConvResidualBlock',
    channels=[
        32,
        64,
        128,
        256,
        512,
    ],
    norm_type='bn',
    num_block_per_stage=2)
norm_type = 'bn'
num_ep = 10
optimizer_config = dict(config=dict(lr=0.001), type='adamw')
resume_ckpt_path = None
run_name = 'lr-1e-3'
test_data_config = dict(
    data_loader_config=dict(batch_size=16, num_workers=4),
    dataset_config=dict(root='DATA'))
train_data_config = dict(
    data_loader_config=dict(batch_size=16, num_workers=4),
    dataset_config=dict(root='DATA'))
trainer_config = dict(
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
    precision='32',
    val_check_interval=0.5)
wandb_config = dict(offline=True, project='backbone-exp')
