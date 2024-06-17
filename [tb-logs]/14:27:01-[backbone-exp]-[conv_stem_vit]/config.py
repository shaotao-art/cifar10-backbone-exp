cifar_data_root = 'DATA'
ckp_config = dict(
    dirpath='06-01-24/14:27:01-[backbone-exp]-[conv_stem_vit]',
    every_n_epochs=None,
    save_last=None)
ckp_root = '06-01-24/14:27:01-[backbone-exp]-[conv_stem_vit]'
device = 'cuda'
enable_wandb = True
img_size = 32
load_weight_from = None
lr_sche_config = dict(config=dict(), type='constant')
model_config = dict(
    conv_channels=128,
    img_size=32,
    patch_size=4,
    torch_transformer_encoder_config=dict(
        layer_config=dict(
            activation='gelu',
            batch_first=True,
            bias=True,
            d_model=128,
            dim_feedforward=512,
            dropout=0.1,
            nhead=4,
            norm_first=True),
        num_layers=11))
num_ep = 100
optimizer_config = dict(config=dict(lr=0.0003), type='adamw')
patch_size = 4
resume_ckpt_path = None
run_name = 'conv_stem_vit'
test_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
train_data_config = dict(
    data_loader_config=dict(batch_size=64, num_workers=4),
    dataset_config=dict(root='DATA'))
trainer_config = dict(
    check_val_every_n_epoch=1, log_every_n_steps=5, precision='32')
wandb_config = dict(offline=True, project='backbone-exp')
