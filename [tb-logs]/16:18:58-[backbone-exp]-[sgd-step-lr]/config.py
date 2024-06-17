act_type = 'relu'
cifar_data_root = 'DATA'
ckp_config = dict(
    dirpath='06-01-24/16:18:58-[backbone-exp]-[sgd-step-lr]',
    every_n_epochs=None,
    save_last=None)
ckp_root = '06-01-24/16:18:58-[backbone-exp]-[sgd-step-lr]'
device = 'cuda'
enable_wandb = True
load_weight_from = None
lr_sche_config = dict(
    config=dict(epoches=[
        60,
        80,
    ], muls=[
        0.1,
        0.1,
    ]), type='step')
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
optimizer_config = dict(config=dict(lr=0.01, momentum=0.9), type='sgd')
resume_ckpt_path = None
run_name = 'sgd-step-lr'
test_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
train_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
trainer_config = dict(
    check_val_every_n_epoch=1, log_every_n_steps=5, precision='32')
wandb_config = dict(offline=True, project='backbone-exp')
