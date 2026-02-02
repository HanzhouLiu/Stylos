# Prepare Dataset

1. **Transfer Generator to Artistic Domain**
   To adapt a pre-trained photo-realistic StyleGAN2 generator to a target artistic domain, refer to [FreezeG](https://github.com/bryandlee/FreezeG) and [Few-Shot GAN Adaptation](https://github.com/WisconsinAIVision/few-shot-gan-adaptation).

2. **Generate Interpolation Series**
   Clone [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), place the provided `generate_samples.py` script inside the cloned repository, and run the command below to:

   - Project source images into latent space
   - Generate interpolation series between source and target domains

   **Note**: The source model should match the domain of the input images (e.g., use a photo-realistic generator for photo-realistic images).

   ```bash
   python generate_samples.py \
       --source_pth /path_to_source_ckpt.pt \
       --target_pth /path_to_target_ckpt.pt \
       --files /path_to_source_images \
       --save_dir ./output
   ```

3. **Prepare Training Metadata File**
   Create a CSV file similar to `dataset/train/sampled_10_dataset.csv`, listing the interpolation series to be used for training.