import torch
import torch.nn as nn
import torch.nn.functional as F
from net.PConv import PartialConv

class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.dp = nn.Dropout()

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        h = self.dp(h)
        return h, h_mask


class PConvUNet(nn.Module):
    # def __init__(self, layer_size=4, input_channels=1, upsampling_mode='nearest'):
    def __init__(self, layer_size=4, input_channels=9, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        self.enc_1 = PCBActiv(input_channels, 32, bn=False, sample='down-7')
        self.enc_2 = PCBActiv(32, 64, sample='down-5')
        self.enc_3 = PCBActiv(64, 128, sample='down-5')
        self.enc_4 = PCBActiv(128, 256, sample='down-3')


        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(256, 256, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv(256 + 256, 256, activ='leaky'))
        self.dec_4 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_2 = PCBActiv(64 + 32, 32, activ='leaky')
        self.dec_1 = PCBActiv(32 + input_channels, 1,
                              bn=False, activ=None, conv_bias=True)


    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        # h_dict['h_0'] 输入input
        # h_mask_dict['h_0'] 输入的mask
        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        # print(input.shape, input_mask.shape)
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            # print(h_key, h_dict[h_key_prev].shape)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            # print("encoder", h_dict[h_key].shape, h_mask_dict[h_key].shape)
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)
            # print("decoder")
            # print(h.shape)
            # print("h", h.shape)
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            # print("up", h.shape)
            # print(h.shape)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='nearest')
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            # print(h.shape)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            # print(h_mask)
            # print(h_key, h_dict[h_key_prev].shape)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)
            # print("out", h.shape)
            # print(h.shape)
            # print("--------------")

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()