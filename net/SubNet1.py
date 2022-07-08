import importlib
import torch.nn as nn
import torch
# from net.ConvLSTM import LSTM_cell
# from net.ConvGRU import GRU_cell
import numpy as np
from net.PConv import PartialConv
import torch.nn.functional as F


class EDStage(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, stage):
        super(EDStage, self).__init__()
        self.stage = stage
        if stage == "encoder":
            self.conv = PartialConv(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        elif stage == "decoder":
            self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(k, k), stride=(s, s), padding=(p, p))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, inputs, mask = None):
        if self.stage == "encoder":
            output, new_mask = self.conv(inputs, mask)
            output = self.leaky_relu(output)
            return output, new_mask
        elif self.stage == "decoder":
            # output = self.leaky_relu(self.conv(inputs))
            output = F.interpolate(inputs, scale_factor=2, mode='nearest')
            new_mask = F.interpolate(mask, scale_factor=2, mode='nearest')
            return output, new_mask

class ConvRNNNet(nn.Module):
    def __init__(self, matrix_len, rnn_type):
        super(ConvRNNNet, self).__init__()

        if rnn_type == "lstm":
            module = importlib.import_module("net.ConvLSTM")
            RNN_cell = getattr(module, "LSTM_cell")
        elif rnn_type == "gru":
            module = importlib.import_module("net.ConvGRU")
            RNN_cell = getattr(module, "GRU_cell")

        # stage1
        self.encoder_stage1 = EDStage(1, 8, 3, 1, 1, "encoder")

        # rnn1
        self.encoder_rnn1 = RNN_cell(shape=(matrix_len, matrix_len), input_channels=8, filter_size=5, num_features=32)
        self.encoder_bn1 = nn.BatchNorm2d(32)

        # stage2
        self.encoder_stage2 = EDStage(32, 32, 3, 2, 1, "encoder")

        # rnn2
        self.encoder_rnn2 = RNN_cell(shape=(matrix_len//2,matrix_len//2), input_channels=32, filter_size=5, num_features=64)
        self.encoder_bn2 = nn.BatchNorm2d(64)

        # stage3
        self.encoder_stage3 = EDStage(64, 64, 3, 2, 1, "encoder")

        # rnn3
        self.encoder_rnn3 = RNN_cell(shape=(matrix_len//4,matrix_len//4), input_channels=64, filter_size=5, num_features=64)
        self.encoder_bn3 = nn.BatchNorm2d(64)

        # rnn3
        self.decoder_rnn3 = RNN_cell(shape=(matrix_len//4,matrix_len//4), input_channels=64, filter_size=5, num_features=64)


        # stage3
        self.decoder_stage3 = EDStage(64, 64, 4, 2, 1, "decoder")
        self.decoder_bn3 = nn.BatchNorm2d(64)


        # rnn2
        self.decoder_rnn2 = RNN_cell(shape=(matrix_len//2,matrix_len//2), input_channels=64, filter_size=5, num_features=64)


        # stage2
        self.decoder_stage2 = EDStage(64, 64, 4, 2, 1, "decoder")
        self.decoder_bn2 = nn.BatchNorm2d(64)


        # rnn1
        self.decoder_rnn1 = RNN_cell(shape=(matrix_len,matrix_len), input_channels=64, filter_size=5, num_features=32)

        # stage1
        self.decoder_stage1_conv1 = PartialConv(32, 8, 3, 1, 1)
        self.decoder_stage1_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.decoder_stage1_conv2 = PartialConv(8, 1, 3, 1, 1)
        self.decoder_stage1_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.decoder_bn1 = nn.BatchNorm2d(64)

        self.dp = nn.Dropout2d(p=0.2)

    def encoder_forward_by_stage(self, inputs, subnet, rnn, mask):
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs, new_mask = subnet(inputs, mask)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))
        new_mask = torch.reshape(new_mask, inputs.shape)
        outputs_stage, state_stage, new_mask = rnn(inputs=inputs, hidden_state=None, mask=new_mask, stage="encoder")
        return outputs_stage, state_stage, new_mask

    def decoder_forward_by_stage(self, inputs, state, subnet, rnn, mask, hidden_mask):
        inputs, state_stage, new_mask = rnn(inputs=inputs, hidden_state=state, mask=mask, hidden_mask=hidden_mask, stage="decoder")
        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        new_mask = torch.reshape(new_mask, inputs.shape)
        inputs, new_mask = subnet(inputs, new_mask)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))
        new_mask = torch.reshape(new_mask, inputs.shape)
        return inputs, new_mask


    def build_bn(self, bn_layer, inputs):
        inputs = torch.cat(torch.chunk(inputs, 9, dim=0), dim=1).squeeze(0)
        inputs = bn_layer(inputs)
        inputs = torch.cat(torch.chunk(inputs.unsqueeze(0), chunks=9, dim=1), dim=0)
        return inputs

    def forward(self, inputs, mask):
        # Encoder
        hidden_states = []
        hidden_masks = []
        inputs = inputs.transpose(0, 1)  # S, B, 1, 50, 50
        seq_number, batch_size, input_channel, height, width = inputs.shape
        mask = torch.cat([mask for _ in range(seq_number)], 0)

        inputs, state_stage, new_mask = self.encoder_forward_by_stage(inputs, self.encoder_stage1, self.encoder_rnn1, mask)

        inputs = self.build_bn(self.encoder_bn1, inputs)
        inputs = self.dp(inputs)

        hidden_states.append(state_stage)
        hidden_masks.append(new_mask)

        seq_number, batch_size, input_channel, height, width = inputs.shape
        new_mask = torch.cat([new_mask for _ in range(int(inputs.shape[2] / new_mask.shape[2]))], 2)
        new_mask = new_mask.view(-1, input_channel, height, width)

        inputs, state_stage, new_mask = self.encoder_forward_by_stage(inputs, self.encoder_stage2, self.encoder_rnn2, new_mask)

        inputs = self.build_bn(self.encoder_bn2, inputs)
        inputs = self.dp(inputs)

        hidden_states.append(state_stage)
        hidden_masks.append(new_mask)

        seq_number, batch_size, input_channel, height, width = inputs.shape
        new_mask = torch.cat([new_mask for _ in range(int(inputs.shape[2] / new_mask.shape[2]))], 2)
        new_mask = new_mask.view(-1, input_channel, height, width)

        inputs, state_stage, new_mask = self.encoder_forward_by_stage(inputs, self.encoder_stage3, self.encoder_rnn3, new_mask)

        inputs = self.build_bn(self.encoder_bn3, inputs)
        inputs = self.dp(inputs)


        hidden_states.append(state_stage)
        hidden_masks.append(new_mask)
        
        
        
        # Decoder
        inputs, new_mask = self.decoder_forward_by_stage(None, hidden_states[2], self.decoder_stage3, self.decoder_rnn3, new_mask, hidden_masks[2])

        inputs = self.build_bn(self.decoder_bn3, inputs)
        inputs = self.dp(inputs)

        inputs, new_mask = self.decoder_forward_by_stage(inputs, hidden_states[1], self.decoder_stage2, self.decoder_rnn2, new_mask, hidden_masks[1])
  
        inputs = self.build_bn(self.decoder_bn2, inputs)
        inputs = self.dp(inputs)

        inputs, state_stage, new_mask = self.decoder_rnn1(inputs=inputs, hidden_state=hidden_states[0], mask=new_mask, hidden_mask=hidden_masks[0], stage="decoder")

        seq_number, batch_size, input_channel, height, width = inputs.shape
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        new_mask = torch.reshape(new_mask, inputs.shape)

        inputs, new_mask = self.decoder_stage1_conv1(inputs, new_mask)

        inputs = self.decoder_stage1_relu1(inputs)

        inputs, new_mask = self.decoder_stage1_conv2(inputs, new_mask)

        # 仅测试sub1需注释掉这一句
        inputs = self.decoder_stage1_relu2(inputs)

        # inputs = F.interpolate(inputs, scale_factor=2, mode='nearest')
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1), inputs.size(2), inputs.size(3)))


        inputs = inputs.transpose(0, 1)
        output = torch.mean(inputs, dim=1)

        return output


if __name__ == '__main__':
    # t = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    # print(t(torch.Tensor(np.zeros((96, 96, 13, 13)))).shape)
    t = torch.Tensor(np.zeros((9, 10, 16, 50, 50)))
    k = torch.Tensor(np.zeros((9, 10, 64, 50, 50)))
    print(torch.cat((t, k), 2).shape)