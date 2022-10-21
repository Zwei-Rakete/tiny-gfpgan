import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import default_init_weights
import math
import random

#----------------------------------------图像转换模块--------------------------------------
class ToRGB(nn.Module):
    """To RGB (image space) from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
    """

    def __init__(self, in_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.interpolate = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=False)

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.interpolate(skip)
            out = out + skip
        return out
#----------------------------------------网络结构模块--------------------------------------
class ResBlock(nn.Module):
    """Residual block with bilinear upsampling/downsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode. Options: down | up. Default: down.
    """

    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2
        self.interpolate1 = nn.Upsample(scale_factor=self.scale_factor,mode='bilinear', align_corners=False)
        self.interpolate2 = nn.Upsample(scale_factor=self.scale_factor,mode='bilinear', align_corners=False)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self, x):
        out = self.leaky_relu1(self.conv1(x))
        # upsample/downsample
        out = self.interpolate1(out)
        out = self.leaky_relu2(self.conv2(out))
        # skip
        x = self.interpolate2(x)
        skip = self.skip(x)
        out = out + skip
        return out

class ModulatedConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        # modulation inside each modulated conv
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        # initialization
        default_init_weights(self.modulation, scale=1, bias_fill=1, a=0, mode='fan_in', nonlinearity='linear')

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size**2))
        self.padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding, bias=False)
        self.conv2d.weight.data = self.weight.view(1 * self.out_channels, self.in_channels,self.kernel_size,self.kernel_size)
        self.interpolate1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.interpolate2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)


    def forward(self, x, style):
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, c, 1, 1)
        x = x * style
        if self.demodulate:
            if self.sample_mode == 'upsample':
                x = self.interpolate1(x)
            elif self.sample_mode == 'downsample':
                x = self.interpolate2(x)
            x = self.conv2d(x)
            weight = self.weight * style
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            out = x * demod.view(b, self.out_channels, 1, 1)
        else:
            if self.sample_mode == 'upsample':
                x = self.interpolate1(x)
            elif self.sample_mode == 'downsample':
                x = self.interpolate2(x)
            x = x.view(1, b * c, h, w)
            out = self.conv2d(x)

        out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})')

class StyleConv(nn.Module):
    """Style conv used in StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None. Default: None.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style) * 2**0.5  # for conversion
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # add bias
        out = out + self.bias
        # activation
        out = self.activate(out)
        return out

class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.num_channel = num_channel
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, x):
        # covert self.weight from constant to dynamic, then self.weight can be next module's input
        x = x.view(1, self.num_channel, 1, 1)
        out = x + self.weight
        out = out - x
        return out


#---------------------------------------Clean--------------------------------------------------------------

class StyleGAN2GeneratorClean(nn.Module):
    """Clean version of StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1):
        super(StyleGAN2GeneratorClean, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.extend(
                [nn.Linear(num_style_feat, num_style_feat, bias=True),
                 nn.LeakyReLU(negative_slope=0.2, inplace=True)])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        # initialization
        #default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        # channel list
        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None)
        self.to_rgb1 = ToRGB(channels['4'], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        #self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            #self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample'))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels


    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorClean.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

#-----------------------------------------------CSFT-------------------------------------------------


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1, sft_half=False):
        super(StyleGAN2GeneratorCSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=False,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorCSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [None] * self.num_layers
                #noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent[:, 0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

#--------------------------------------------GFPGAN-------------------------------------------
class GFPGANv1Clean(nn.Module):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(GFPGANv1Clean, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat
        #self.input = Input()
        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = nn.Conv2d(3, channels[f'{first_out_size}'], 1)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f'{2**i}'], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)


        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=False):
        """Forward function for GFPGANv1Clean.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []
        
        # encoder
        feat = self.leaky_relu1(self.conv_body_first(x))
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = self.leaky_relu2(self.final_conv(feat))

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs