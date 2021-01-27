import torch
from torch import nn as nn

from torch.nn import functional as F

from basicsr.models.archs.arch_util import (ResidualBlockNoBN, Upsample, UBlock,
                                            make_layer)


class EDUSR(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDUSR, self).__init__()

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = UBlock(num_feat)
        
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        _,_,h,w = x.size()
        assert h % 16 ==0 and w%16 == 0, ('The hieght and width must be multiple of 16.')

        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)

        x, o1,o2,o3,o4 = self.body(x) 

        o1 = F.interpolate(o1, scale_factor=2, mode='bilinear') / self.img_range + self.mean
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear') / self.img_range + self.mean
        o3 = F.interpolate(o3, scale_factor=8, mode='bilinear') / self.img_range + self.mean
        o4 = F.interpolate(o4, scale_factor=16, mode='bilinear') / self.img_range + self.mean

        res = self.conv_after_body(x)
        res += x

        x = self.conv_last(self.upsample(res))

        x = x / self.img_range + self.mean

        return torch.cat((x, o1, o2, o3, o4), 1)