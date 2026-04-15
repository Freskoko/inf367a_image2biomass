# run from inf367a_image2biomass/
ROOT_PATH="."
DATA_PATH="src/data"

SEED=2026
NITERS=1000 # limit
BATCH_SIZE_D=32
BATCH_SIZE_G=512
START_COUNT=1 # 1.04
END_COUNT=185 # 185.7
STEPSIZE_COUNT=2
N_IMGS_PER_CELLCOUNT=10
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4

# fid stuff
NFAKE_PER_LABEL=40
FID_RADIUS=20

# cgan
DIM_CcGAN=256
DIM_cGAN=128
DIM_EMBED=128
LOSS_TYPE='vanilla'

COMP_FID="False"

cd $ROOT_PATH

echo "-------------------------------------------------------------------------------------------------"
echo "PREPROCESS DATA"
python src/main/ccgan_improved/preprocess/preprocess.py


echo "-------------------------------------------------------------------------------------------------"
echo "AE"
python src/main/ccgan_improved/pretrain/pretrain_AE.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --dim_bottleneck 512 \
    --epochs 50 \
    --resume_epoch 0 \
    --save_ckpt_freq 25 \
    --batch_size_train 256 \
    --batch_size_valid 128 \
    --base_lr 1e-4 \
    --lr_decay_epochs 25 \
    --lr_decay_factor 0.5 \
    --weight_dacay 1e-5 \
    --seed $SEED \
    --min_label $START_COUNT \
    --max_label $END_COUNT

echo "-------------------------------------------------------------------------------------------------"
echo "regression CNN for evaluation"
python src/main/ccgan_improved/pretrain/pretrain_CNN_regre.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --start_count $START_COUNT \
    --end_count $END_COUNT \
    --CNN ResNet34_regre \
    --epochs 200 \
    --batch_size_train 256 \
    --batch_size_valid 64 \
    --base_lr 0.01 \
    --seed $SEED \
    --transform

N_CLASS=50 # 100
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS"
python src/main/ccgan_improved/run.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --GAN cGAN \
    --cGAN_num_classes $N_CLASS \
    --start_count $START_COUNT \
    --end_count $END_COUNT \
    --stepsize_count $STEPSIZE_COUNT \
    --num_imgs_per_count $N_IMGS_PER_CELLCOUNT \
    --transform \
    --niters_gan $NITERS \
    --resume_niters_gan 0 \
    --save_niters_freq 2000 \
    --dim_gan $DIM_cGAN \
    --lr_g_gan $LR_G \
    --lr_d_gan $LR_D \
    --batch_size_disc $BATCH_SIZE_D \
    --batch_size_gene $BATCH_SIZE_G \
    --seed $SEED \
    --visualize_fake_images \
    --comp_FID $COMP_FID \
    --nfake_per_label $NFAKE_PER_LABEL \
    --samp_batch_size 1000 \
    --FID_radius $FID_RADIUS \
    --dump_fake_for_NIQE

RESUME_NITER=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Hard"
python src/main/ccgan_improved/run.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --GAN CcGAN \
    --start_count $START_COUNT \
    --end_count $END_COUNT \
    --stepsize_count $STEPSIZE_COUNT \
    --num_imgs_per_count $N_IMGS_PER_CELLCOUNT \
    --transform \
    --kernel_sigma $SIGMA \
    --threshold_type hard \
    --kappa $KAPPA \
    --dim_gan $DIM_CcGAN \
    --loss_type_gan $LOSS_TYPE \
    --niters_gan $NITERS \
    --resume_niters_gan 0 \
    --save_niters_freq 2000 \
    --lr_g_gan $LR_G \
    --lr_d_gan $LR_D \
    --batch_size_disc $BATCH_SIZE_D \
    --batch_size_gene $BATCH_SIZE_G \
    --seed $SEED \
    --visualize_fake_images \
    --dim_embed $DIM_EMBED \
    --comp_FID $COMP_FID \
    --nfake_per_label $NFAKE_PER_LABEL \
    --samp_batch_size 1000 \
    --FID_radius $FID_RADIUS \
    --dump_fake_for_NIQE \
    --num_channels 3

RESUME_NITER=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Soft"
python src/main/ccgan_improved/run.py \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --GAN CcGAN \
    --start_count $START_COUNT \
    --end_count $END_COUNT \
    --stepsize_count $STEPSIZE_COUNT \
    --num_imgs_per_count $N_IMGS_PER_CELLCOUNT \
    --transform \
    --kernel_sigma $SIGMA \
    --threshold_type soft \
    --kappa $KAPPA \
    --dim_gan $DIM_CcGAN \
    --loss_type_gan $LOSS_TYPE \
    --niters_gan $NITERS \
    --resume_niters_gan 0 \
    --save_niters_freq 2000 \
    --lr_g_gan $LR_G \
    --lr_d_gan $LR_D \
    --batch_size_disc $BATCH_SIZE_D \
    --batch_size_gene $BATCH_SIZE_G \
    --seed $SEED \
    --visualize_fake_images \
    --dim_embed $DIM_EMBED \
    --comp_FID $COMP_FID \
    --nfake_per_label $NFAKE_PER_LABEL \
    --samp_batch_size 1000 \
    --FID_radius $FID_RADIUS \
    --dump_fake_for_NIQE \
    --num_channels 3