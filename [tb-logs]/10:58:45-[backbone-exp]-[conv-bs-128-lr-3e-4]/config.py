act_type = 'relu'
cifar_data_root = 'DATA'
ckp_config = dict(
    dirpath='06-01-24/10:58:45-[backbone-exp]-[conv-bs-128-lr-3e-4]',
    every_n_epochs=None,
    save_last=None)
ckp_root = '06-01-24/10:58:45-[backbone-exp]-[conv-bs-128-lr-3e-4]'
device = 'cuda'
enable_wandb = True
load_weight_from = None
lr_sche_config = dict(config=dict(), type='constant')
model_config = dict(
    act_type='relu',
    base_block_config=dict(
        act='relu',
        in_channels=256,
        norm_config=dict(config=dict(num_features=256), type='bn'),
        out_channels=256),
    block_type='TwoConvBlock',
    channels=[
        32,
        64,
        128,
        256,
        256,
    ],
    norm_type='bn',
    num_block_per_stage=1)
norm_type = 'bn'
num_ep = 100
optimizer_config = dict(config=dict(lr=0.0003), type='adamw')
resume_ckpt_path = None
run_name = 'conv-bs-128-lr-3e-4'
test_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
train_data_config = dict(
    data_loader_config=dict(batch_size=128, num_workers=4),
    dataset_config=dict(root='DATA'))
trainer_config = dict(
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
    precision='32',
    val_check_interval=0.5)
wandb_config = dict(offline=True, project='backbone-exp')
