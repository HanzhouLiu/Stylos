"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import copy
from dataclasses import dataclass
from functools import partial
import time
import warnings

import numpy as np
from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF














def gen_scales(start, end):
    scale = end
    i = 0
    scales = set()
    while scale >= start:
        scales.add(scale)
        i += 1
        scale = round(end / pow(2, i/2))
    return sorted(scales)


def interpolate(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return F.interpolate(*args, **kwargs)


def scale_adam(state, shape):
    """Prepares a state dict to warm-start the Adam optimizer at a new scale."""
    state = copy.deepcopy(state)
    for group in state['state'].values():
        exp_avg, exp_avg_sq = group['exp_avg'], group['exp_avg_sq']
        group['exp_avg'] = interpolate(exp_avg, shape, mode='bicubic')
        group['exp_avg_sq'] = interpolate(exp_avg_sq, shape, mode='bilinear').relu_()
        if 'max_exp_avg_sq' in group:
            max_exp_avg_sq = group['max_exp_avg_sq']
            group['max_exp_avg_sq'] = interpolate(max_exp_avg_sq, shape, mode='bilinear').relu_()
    return state


@dataclass
class STIterate:
    w: int
    h: int
    i: int
    i_max: int
    loss: float
    time: float
    gpu_ram: int


class StyleTransfer:
    def __init__(self, devices=['cpu'], pooling='max'):
        self.devices = [torch.device(device) for device in devices]
        self.image = None
        self.average = None

        # The default content and style layers follow Gatys et al. (2015).
        self.content_layers = [22]
        self.style_layers = [1, 6, 11, 20, 29]

        # The weighting of the style layers differs from Gatys et al. (2015) and Johnson et al.
        style_weights = [256, 64, 16, 4, 1]
        weight_sum = sum(abs(w) for w in style_weights)
        self.style_weights = [w / weight_sum for w in style_weights]

        self.model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling)

        if len(self.devices) == 1:
            device_plan = {0: self.devices[0]}
        elif len(self.devices) == 2:
            device_plan = {0: self.devices[0], 5: self.devices[1]}
        else:
            raise ValueError('Only 1 or 2 devices are supported.')

        self.model.distribute_layers(device_plan)

    def get_image_tensor(self):
        return self.average.get().detach()[0].clamp(0, 1)

    def get_image(self, image_type='pil'):
        if self.average is not None:
            image = self.get_image_tensor()
            if image_type.lower() == 'pil':
                return TF.to_pil_image(image)
            elif image_type.lower() == 'np_uint16':
                arr = image.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))
            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")

    def stylize(self, content_image, style_images, *,
                style_weights=None,
                content_weight: float = 0.015,
                tv_weight: float = 2.,
                optimizer: str = 'adam',
                min_scale: int = 128,
                end_scale: int = 512,
                iterations: int = 500,
                initial_iterations: int = 1000,
                step_size: float = 0.02,
                avg_decay: float = 0.99,
                init: str = 'content',
                style_scale_fac: float = 1.,
                style_size: int = None,
                callback=None):

        min_scale = min(min_scale, end_scale)
        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]
        if len(style_images) != len(style_weights):
            raise ValueError('style_images and style_weights must have the same length')

        tv_loss = Scale(LayerApply(TVLoss(), 'input'), tv_weight)

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
        elif init == 'gray':
            self.image = torch.rand([1, 3, ch, cw]) / 255 + 0.5
        elif init == 'uniform':
            self.image = torch.rand([1, 3, ch, cw])
        elif init == 'normal':
            self.image = torch.empty([1, 3, ch, cw])
            nn.init.trunc_normal_(self.image, mean=0.5, std=0.25, a=0, b=1)
        elif init == 'style_stats':
            means, variances = [], []
            for i, image in enumerate(style_images):
                my_image = TF.to_tensor(image)
                means.append(my_image.mean(dim=(1, 2)) * style_weights[i])
                variances.append(my_image.var(dim=(1, 2)) * style_weights[i])
            means = sum(means)
            variances = sum(variances)
            channels = []
            for mean, variance in zip(means, variances):
                channel = torch.empty([1, 1, ch, cw])
                nn.init.trunc_normal_(channel, mean=mean, std=variance.sqrt(), a=0, b=1)
                channels.append(channel)
            self.image = torch.cat(channels, dim=1)
        else:
            raise ValueError("init must be one of 'content', 'gray', 'uniform', 'style_mean'")
        self.image = self.image.to(self.devices[0])

        opt = None

        # Stylize the image at successively finer scales, each greater by a factor of sqrt(2).
        # This differs from the scheme given in Gatys et al. (2016).
        for scale in scales:
            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache()

            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            content = TF.to_tensor(content_image.resize((cw, ch), Image.BICUBIC))[None]
            content = content.to(self.devices[0])

            self.image = interpolate(self.image.detach(), (ch, cw), mode='bicubic').clamp(0, 1)
            self.average = EMA(self.image, avg_decay)
            self.image.requires_grad_()

            print(f'Processing content image ({cw}x{ch})...')
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for layer, weight in zip(self.content_layers, content_weights):
                target = content_feats[layer]
                content_losses.append(Scale(LayerApply(ContentLossMSE(target), layer), weight))

            style_targets, style_losses = {}, []
            for i, image in enumerate(style_images):
                if style_size is None:
                    sw, sh = size_to_fit(image.size, round(scale * style_scale_fac))
                else:
                    sw, sh = size_to_fit(image.size, style_size)
                style = TF.to_tensor(image.resize((sw, sh), Image.BICUBIC))[None]
                style = style.to(self.devices[0])
                print(f'Processing style image ({sw}x{sh})...')
                style_feats = self.model(style, layers=self.style_layers)
                # Take the weighted average of multiple style targets (Gram matrices).
                for layer in self.style_layers:
                    target_mean, target_cov = StyleLossW2.get_target(style_feats[layer])
                    target_mean *= style_weights[i]
                    target_cov *= style_weights[i]
                    if layer not in style_targets:
                        style_targets[layer] = target_mean, target_cov
                    else:
                        style_targets[layer][0].add_(target_mean)
                        style_targets[layer][1].add_(target_cov)
            for layer, weight in zip(self.style_layers, self.style_weights):
                target = style_targets[layer]
                style_losses.append(Scale(LayerApply(StyleLossW2(target), layer), weight))

            crit = SumLoss([*content_losses, *style_losses, tv_loss])

            if optimizer == 'adam':
                opt2 = optim.Adam([self.image], lr=step_size, betas=(0.9, 0.99))
                # Warm-start the Adam optimizer if this is not the first scale.
                if scale != scales[0]:
                    opt_state = scale_adam(opt.state_dict(), (ch, cw))
                    opt2.load_state_dict(opt_state)
                opt = opt2
            elif optimizer == 'lbfgs':
                opt = optim.LBFGS([self.image], max_iter=1, history_size=10)
            else:
                raise ValueError("optimizer must be one of 'adam', 'lbfgs'")

            if self.devices[0].type == 'cuda':
                torch.cuda.empty_cache()

            def closure():
                feats = self.model(self.image)
                loss = crit(feats)
                loss.backward()
                return loss

            actual_its = initial_iterations if scale == scales[0] else iterations
            for i in range(1, actual_its + 1):
                opt.zero_grad()
                loss = opt.step(closure)
                # Enforce box constraints, but not for L-BFGS because it will mess it up.
                if optimizer != 'lbfgs':
                    with torch.no_grad():
                        self.image.clamp_(0, 1)
                self.average.update(self.image)
                if callback is not None:
                    gpu_ram = 0
                    for device in self.devices:
                        if device.type == 'cuda':
                            gpu_ram = max(gpu_ram, torch.cuda.max_memory_allocated(device))
                    callback(STIterate(w=cw, h=ch, i=i, i_max=actual_its, loss=loss.item(),
                                       time=time.time(), gpu_ram=gpu_ram))

            # Initialize each new scale with the previous scale's averaged iterate.
            with torch.no_grad():
                self.image.copy_(self.average.get())

        return self.get_image()