# Training from scratch on a sampled dataset
device_index=0
label_path="./dataset/train/sampled_10_dataset.csv"
backbone_config="resnet50"
epoch=10
batch_size=4
lr=1e-4
loss='listMLE'

echo use_device:${device_index}


CUDA_VISIBLE_DEVICES=${device_index} python3 train.py \
--label_path ${label_path} \
--epoch ${epoch} \
--backbone_config ${backbone_config} \
--batch_size ${batch_size} \
--lr ${lr} \
--loss ${loss} \
--dont_save_model