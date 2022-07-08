import torch
import torch.nn as nn
from net.PConv import PartialConv
# from PConv import PartialConv


class GRU_cell(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(GRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        
        self.conv1 = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=2 * self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm1 = nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features)
        
        self.conv2 = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm2 = nn.GroupNorm(self.num_features // 32, self.num_features)

        self.conv1_backward = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=2 * self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm1_backward = nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features)
        
        self.conv2_backward = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm2_backward = nn.GroupNorm(self.num_features // 32, self.num_features)

    def forward(self, inputs=None, hidden_state=None, seq_len=9, mask=None, hidden_mask=None, stage=None):
        ht_list = []

        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        
        output_inner_forward = []

        forward_new_mask = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]
            
            inner_forward_mask = mask[index, ...]

            if stage == "encoder":
                inner_forward_mask = torch.cat([inner_forward_mask for _ in range(int(self.num_features / self.input_channels) + 1)], 1)
            elif stage == "decoder":
                inner_forward_hidden_mask = hidden_mask[index, ...]
                inner_forward_mask = torch.cat([inner_forward_mask, inner_forward_hidden_mask], 1)


            combined_1 = torch.cat((x, htprev), 1)
            gates, new_mask1 = self.conv1(combined_1, inner_forward_mask)
            gates = self.group_norm1(gates)

            new_mask1, _ = torch.split(new_mask1, self.num_features, dim=1)
            z_gate, rgate = torch.split(gates, self.num_features, dim=1)
            z = torch.sigmoid(z_gate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), 1)
            ht, new_mask2 = self.conv2(combined_2, inner_forward_mask)
            ht = self.group_norm2(ht)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner_forward.append(htnext)
            htprev = htnext
            if index == seq_len // 2:
                ht_list.append(htnext)

            forward_new_mask.append(new_mask1 + new_mask2)

        if hidden_state is None:
            htprev_backward = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            htprev_backward = hidden_state
        output_inner_backward = []
        backward_new_mask = []
        for index in list(range(seq_len))[::-1]:
            if inputs is None:
                x_backward = torch.zeros(htprev.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x_backward = inputs[index, ...]
            
            inner_backward_mask = mask[index, ...]

            if stage == "encoder":
                inner_backward_mask = torch.cat([inner_backward_mask for _ in range(int(self.num_features / self.input_channels) + 1)], 1)
            elif stage == "decoder":
                inner_backward_hidden_mask = hidden_mask[index, ...]
                inner_backward_mask = torch.cat([inner_backward_mask, inner_backward_hidden_mask], 1)
            

            combined_1_backward = torch.cat((x_backward, htprev_backward), 1)

            gates_backward, new_mask1 = self.conv1_backward(combined_1_backward, inner_backward_mask)
            gates_backward = self.group_norm1_backward(gates_backward)
            new_mask1, _ = torch.split(new_mask1, self.num_features, dim=1)

            z_gate_backward, rgate_backward = torch.split(gates_backward, self.num_features, dim=1)
            z_backward = torch.sigmoid(z_gate_backward)
            r_backward = torch.sigmoid(rgate_backward)

            combined_2_backward = torch.cat((x_backward, r_backward * htprev_backward), 1)
            ht_backward, new_mask2 = self.conv2_backward(combined_2_backward, inner_backward_mask)
            ht_backward = self.group_norm2_backward(ht_backward)
            ht_backward = torch.tanh(ht_backward)
            htnext_backward = (1 - z_backward) * htprev_backward + z_backward * ht_backward
            output_inner_backward.insert(0, htnext_backward)
            htprev_backward = htnext_backward
            if index == seq_len // 2:
                ht_list.append(htnext_backward)
            
            backward_new_mask.append(new_mask1 + new_mask2)

        output = torch.mean(torch.stack([torch.stack(output_inner_forward), torch.stack(output_inner_backward)]), dim=0)
        htnext = torch.mean(torch.stack(ht_list), dim=0)

        new_mask_list = []
        for forward_mask, backward_mask in zip(forward_new_mask, backward_new_mask[::-1]):
            mask = forward_mask + backward_mask
            mask[mask > 0] = 1
            new_mask_list.append(mask)
        new_mask = torch.stack(new_mask_list)
        return output, htnext, new_mask


if __name__ == '__main__':
    x = torch.randn((9, 1, 8, 64, 64)).cuda()
    mask = torch.ones((9, 1, 8, 64, 64)).cuda()
    encoder_rnn1 = GRU_cell(shape=(64, 64), input_channels=8, filter_size=5, num_features=32).cuda()
    encoder_rnn1(inputs=x, hidden_state=None, mask=mask, stage="encoder")