import shutil
from collections import OrderedDict
try:
    from projector import *
except ImportError:
    raise ImportError("Please prepare stylegan2 repository: https://github.com/rosinality/stylegan2-pytorch")


def project_to_latent(args):
    device = args.device
    n_mean_latent = 10000
    resize = min(args.size, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if os.path.isdir(args.files):
        files = os.listdir(args.files)
        files = [os.path.join(args.files, img_name) for img_name in files]
        print(f"loading imgs from {args.files}")
    elif os.path.isfile(args.files) and args.files.endswith('txt'):
        with open(args.files, 'r') as f:
            files = f.read().splitlines()
    else:
        raise FileNotFoundError

    files = sorted(files)
    files = files[args.start: args.end]

    num_batch = math.ceil(len(files)/args.batch_size)
    for batch_idx in range(num_batch):
        print(f"batch {batch_idx+1} of {num_batch}...")
        cur_files = files[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        cur_imgs = []
        cur_file_names = []
        for imgfile in cur_files:
            try:
                img = transform(Image.open(imgfile).convert("RGB"))
                cur_imgs.append(img)
                cur_file_names.append(imgfile)
            except:
                print(f"unable to load {imgfile}")
        if len(cur_imgs) == 0:
            continue


        imgs = torch.stack(cur_imgs, 0).to(device)
        g_ema = Generator(args.size, 512, 8)
        g_ema.load_state_dict(load_ckpt(args.source_pth), strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)

        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 300 == 0:
                # save results every 300 steps
                save_dirs = [os.path.join(args.save_dir, f"step_{i+1}", get_basename(f)) for f in cur_file_names]
                for f, dst in zip(cur_file_names, save_dirs):
                    os.makedirs(dst, exist_ok=True)
                    shutil.copyfile(f, os.path.join(dst, 'original.png'))

                generate_interpolation(latent_in.detach().clone(), noises, save_dirs, args)

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )


def get_basename(file_pth):
    return os.path.splitext(os.path.basename(file_pth))[0]


def load_ckpt(ckpt_pth):
    state_dict = torch.load(ckpt_pth, weights_only=False)["g_ema"]
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def generate_interpolation(latents, noises, save_dirs, args):
    # ratio: of the source model
    for fuse_ratio in tqdm(range(10, -1, -1)):
        fused_model = fuse(args.source_pth, args.target_pth, ratio=fuse_ratio * 0.1)
        fused_model.eval()
        fused_model.to(args.device)
        img_gen, _ = fused_model([latents], input_is_latent=True, noise=noises)
        img_ar = make_image(img_gen)
        for dir_, img in zip(save_dirs, img_ar):
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(dir_, f'{fuse_ratio}.png'))


def fuse(source_pth, target_pth, layer='last', ratio=1.0, mode='left'):
    """
    Args:
        source_pth: source ckpt path
        target_pth: target ckpt path
        layer: which layers to fuse; first/last/all
        ratio: ratio of the source model to be kept: r*source+(1-r)*target
        mode: whose params to keep for the layers without fusing procedure

    Returns:
        Generator
    """
    gen = Generator(
            256, 512, 8
        )
    state_dict = load_ckpt(source_pth)
    state_dict1 = load_ckpt(source_pth)
    state_dict2 = load_ckpt(target_pth)

    if layer == 'first':
        for i, (k, v1, v2) in enumerate(zip(state_dict.keys(), state_dict1.values(), state_dict2.values())):
            if i <= 58 or 92 <= i <= 106:
                state_dict[k] = ratio*v1.cpu() + (1-ratio)*v2.cpu()
            elif mode == 'left':
                state_dict[k] = v1.cpu()
            else:
                state_dict[k] = v2.cpu()

    elif layer == 'last':
        for i, (k, v1, v2) in enumerate(zip(state_dict.keys(), state_dict1.values(), state_dict2.values())):
            if 92 > i > 58 or i > 106:
                state_dict[k] = ratio*v1.cpu() + (1-ratio)*v2.cpu()
            elif mode == 'left':
                state_dict[k] = v1.cpu()
            else:
                state_dict[k] = v2.cpu()

    else:
        for i, (k, v1, v2) in enumerate(zip(state_dict.keys(), state_dict1.values(), state_dict2.values())):
            state_dict[k] = ratio*v1.cpu() + (1-ratio)*v2.cpu()

    gen.load_state_dict(state_dict, strict=False)
    return gen





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces; then fuse the source and target models"
    )
    parser.add_argument(
        "--source_pth", type=str, required=True, help="path to the source model checkpoint; \
        same domain as the projected images"
    )
    parser.add_argument(
        "--target_pth", type=str, required=True, help="path to the target model checkpoint"
    )
    parser.add_argument(
        "--files", type=str, required=True, help="path to images to be projected \
        or a txt file containg abs path to those images"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="project from start to end; workaround for slurm jobs"
    )
    parser.add_argument(
        "--end", type=int, default=5000, help="project from start to end; workaround for slurm jobs"
    )
    parser.add_argument(
        "--device", type=str, default='cuda:0'
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--save_dir", type=str, help="save dir"
    )
    # kept as defaults from the projector
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1200, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )

    args = parser.parse_args()
    project_to_latent(args)
