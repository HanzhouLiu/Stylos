# This script is used to run inference on a pre-trained model for art score prediction.
python infer_artscore.py \
--auto_parse \
--ckpt_pth ./ckpt/loss@listMLE_model@resnet50_denseLayer@True_batch_size@16_lr@0.0001_dropout@0.5_E_8.pth \
--infer_path ./dataset/eval \
--output_dir ./eval_results \
--output_name result.csv