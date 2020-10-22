import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MultiOctConv.model import MultiOctaveConv
from MultiOctConv.utils import oct_sum

class OctResidualBlock(nn.Module):        
	"""Residual Block with instance normalization."""
	def __init__(self, dim_in, dim_out, alpha, beta, upsample, downsample):
		super(OctResidualBlock, self).__init__()
		self.main = nn.Sequential(
								MultiOctaveConv(dim_in, dim_out, kernel_size = 3,
								alpha_in = alpha, alpha_out = alpha, beta_in = beta, beta_out = beta,
								conv_args = {"stride":1, "padding":1, "bias":False},
								activation_function = nn.ReLU, activation_function_args = {"inplace":True},
								norm_layer = nn.InstanceNorm2d,
								norm_layer_args = {"affine":True, "track_running_stats":True},
								downsample = downsample, upsample = upsample
								),
								MultiOctaveConv(dim_out, dim_out, kernel_size = 3,
								alpha_in = alpha, alpha_out = alpha, beta_in = beta, beta_out = beta,
								conv_args = {"stride":1, "padding":1, "bias":False},
								norm_layer = nn.InstanceNorm2d,
								norm_layer_args = {"affine":True, "track_running_stats":True},
								downsample = downsample, upsample = upsample
								)
					)
	def forward(self, x):
		return oct_sum(x, self.main(x))


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, 
                alpha=0.5, beta=0.0,
                downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2), upsample=nn.Upsample(scale_factor=2, mode='nearest')
                ):
        super(Generator, self).__init__()

        layers = []
        layers.append(MultiOctaveConv(3+c_dim, conv_dim, kernel_size=7,
                                    alpha_in = 0, alpha_out = alpha, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":1, "padding":3, "bias":False},
                                    activation_function = nn.ReLU, activation_function_args = {"inplace":True},
                                    norm_layer = nn.InstanceNorm2d,
                                    norm_layer_args = {"affine":True, "track_running_stats":True},
                                    downsample = downsample, upsample = upsample
                                    )
        )
        ## Down-sampling layers.
        curr_dim = conv_dim
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4, 
                                    alpha_in = alpha, alpha_out = alpha, beta_in = 0, beta_out = beta,
                                    conv_args ={"stride":2, "padding":1, "bias":False},
                                    activation_function = nn.ReLU, activation_function_args = {"inplace":True},
                                    norm_layer = nn.InstanceNorm2d,
                                    norm_layer_args = {"affine":True, "track_running_stats":True},
                                    downsample = downsample, upsample = upsample
                                    )
        )
        curr_dim = curr_dim * 2
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4, 
                                    alpha_in = alpha, alpha_out = alpha, beta_in = beta, beta_out = beta,
                                    conv_args = {"stride":2, "padding":1, "bias":False},
                                    activation_function = nn.ReLU, activation_function_args = {"inplace":True},
                                    norm_layer = nn.InstanceNorm2d,
                                    norm_layer_args = {"affine":True, "track_running_stats":True},
                                    downsample = downsample, upsample = upsample
                                    )
        )
        curr_dim = curr_dim * 2

        ## Bottleneck layers.
        for i in range(repeat_num):
            layers.append(OctResidualBlock(dim_in=curr_dim, dim_out=curr_dim, 
                                            alpha=alpha, beta=beta, upsample=upsample, downsample=downsample))

        layers.append(MultiOctaveConv(curr_dim, curr_dim//2, kernel_size=4,
                                alpha_in = alpha, alpha_out = alpha, beta_in = beta, beta_out = beta,
                                conv_args = {"stride":2, "padding":1, "bias":False},
                                activation_function = nn.ReLU, activation_function_args = {"inplace":True},
                                norm_layer = nn.InstanceNorm2d,
                                norm_layer_args = {"affine":True, "track_running_stats":True},
                                downsample = downsample, upsample = upsample,
                                conv = nn.ConvTranspose2d
                                )
        )
        curr_dim = curr_dim // 2
        layers.append(MultiOctaveConv(curr_dim, curr_dim//2, kernel_size=4, 
                                alpha_in = alpha, alpha_out = alpha + beta, beta_in = beta, beta_out = 0.,
                                conv_args = {"stride":2, "padding":1, "bias":False},
                                activation_function = nn.ReLU, activation_function_args = {"inplace":True},
                                norm_layer = nn.InstanceNorm2d,
                                norm_layer_args = {"affine":True, "track_running_stats":True},
                                downsample = downsample, upsample = upsample,
                                conv = nn.ConvTranspose2d
                                )
        )
        curr_dim = curr_dim // 2
        alpha += beta
        layers.append(MultiOctaveConv(curr_dim, 3, kernel_size=7,
                                    alpha_in = alpha, alpha_out = 0.0, beta_in = 0.0, beta_out = 0.0,
                                    conv_args = {"stride":1, "padding":3, "bias":False},
                                    activation_function = nn.Tanh, downsample = downsample, upsample = upsample
                                    )
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)[0]

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, alpha=0.5, beta=0.,
                downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2), upsample=nn.Upsample(scale_factor=2, mode='nearest')
                ):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(MultiOctaveConv(3, conv_dim, kernel_size=4,
                                    alpha_in = 0, alpha_out = alpha, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
        )
        curr_dim = conv_dim
        layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4,
                                    alpha_in = alpha, alpha_out = alpha, beta_in = 0, beta_out = beta,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
            )
        curr_dim = curr_dim * 2
        flag1 = True
        flag2 = True
        for i in range(2, repeat_num):

            if( image_size//(2**( i + 2 ) ) > 3 ):
                layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4,
                                    alpha_in = alpha, alpha_out = alpha, beta_in = beta, beta_out = beta,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
                )

            elif ( flag1 and (  image_size//(2**( i + 2 ) ) < 3 or i == repeat_num -1) ):
                layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4,
                                    alpha_in = alpha, alpha_out = alpha+beta, beta_in = beta, beta_out = 0,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
                )
                alpha += beta
                flag1 = False

            elif flag2:
                layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4,
                                    alpha_in = alpha, alpha_out = 0, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
                )
                flag2 = False

            else:
                layers.append(MultiOctaveConv(curr_dim, curr_dim*2, kernel_size=4,
                                    alpha_in = 0, alpha_out = 0, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":2, "padding":1},
                                    activation_function = nn.LeakyReLU, activation_function_args = {"negative_slope": 0.01},
                                    downsample = downsample, upsample = upsample
                                    )
                )

            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        if flag2:
            self.conv1 =MultiOctaveConv(curr_dim, 1, kernel_size=3,
                                    alpha_in = alpha, alpha_out = 0, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":1, "padding":1, "bias":False},
                                    downsample = downsample, upsample = upsample
                                    )
            self.conv2 =MultiOctaveConv(curr_dim, c_dim, kernel_size=(kernel_size, kernel_size//2),
                                        alpha_in = alpha, alpha_out = 0, beta_in = 0, beta_out = 0,
                                        conv_args ={"bias":False},
                                        )

        else:
            self.conv2 =MultiOctaveConv(curr_dim, c_dim, kernel_size=kernel_size,
                                        alpha_in = 0, alpha_out = 0, beta_in = 0, beta_out = 0,
                                        conv_args ={"bias":False},
                                        )
            self.conv1 =MultiOctaveConv(curr_dim, 1, kernel_size=3,
                                    alpha_in = 0, alpha_out = 0, beta_in = 0, beta_out = 0,
                                    conv_args ={"stride":1, "padding":1, "bias":False},
                                    downsample = downsample, upsample = upsample
                                    )

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)[0]
        out_cls = self.conv2(h)[0]
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
