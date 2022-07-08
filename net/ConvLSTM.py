import torch
import torch.nn as nn
from net.PConv import PartialConv

class LSTM_cell(nn.Module):
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(LSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2

        self.conv = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=4 * self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm = nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
        
       
        self.conv_backward = PartialConv(in_channels=self.input_channels + self.num_features, out_channels=4 * self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.group_norm_backward = nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
        


    def forward(self, inputs=None, hidden_state=None, seq_len=9, mask=None, hidden_mask=None, stage=None):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            hx, cx = hidden_state

        ht_list = []
        output_inner_forward = []
        
        forward_new_mask = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]
            
            inner_forward_mask = mask[index, ...]
            
            if stage == "encoder":
                inner_forward_mask = torch.cat([inner_forward_mask for _ in range(int(self.num_features / self.input_channels) + 1)], 1)
            elif stage == "decoder":
                inner_forward_hidden_mask = hidden_mask[index, ...]
                inner_forward_mask = torch.cat([inner_forward_mask, inner_forward_hidden_mask], 1)
                
            
            combined1 = torch.cat((x, hx), 1)
            gates, new_mask = self.conv(combined1, inner_forward_mask)  # gates: S, num_features*4, H, W
            gates = self.group_norm(gates)
            
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            new_mask, _, _, _ = torch.split(new_mask, self.num_features, dim=1)
            forward_new_mask.append(new_mask)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner_forward.append(hy)
            hx = hy
            cx = cy
            if index == seq_len // 2:
                ht_list.append((hy, cy))



        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        
        output_inner_backward = []
        backward_new_mask = []
        for index in range(seq_len)[::-1]:
            if inputs is None:
                x_backward = torch.zeros(hx.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x_backward = inputs[index, ...]

            inner_backward_mask = mask[index, ...]
            if stage == "encoder":
                inner_backward_mask = torch.cat([inner_backward_mask for _ in range(int(self.num_features / self.input_channels) + 1)], 1)
            elif stage == "decoder":
                inner_backward_hidden_mask = hidden_mask[index, ...]
                inner_backward_mask = torch.cat([inner_backward_mask, inner_backward_hidden_mask], 1)
            

            combined1 = torch.cat((x_backward, hx), 1)
            gates, new_mask = self.conv_backward(combined1, inner_backward_mask)  # gates: S, num_features*4, H, W

            gates = self.group_norm_backward(gates)
            
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            new_mask, _, _, _ = torch.split(new_mask, self.num_features, dim=1)
            backward_new_mask.append(new_mask)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner_backward.append(hy)
            hx = hy
            cx = cy
            if index == seq_len // 2:
                ht_list.append((hy, cy))

        new_mask_list = []
        for forward_mask, backward_mask in zip(forward_new_mask, backward_new_mask[::-1]):
            mask = forward_mask + backward_mask
            mask[mask > 0] = 1
            new_mask_list.append(mask)
        new_mask = torch.stack(new_mask_list)

        hy_forward, cy_forward = ht_list[0]
        hy_backward, cy_backward = ht_list[1]
        hy = torch.mean(torch.stack([hy_forward, hy_backward]), dim=0)
        cy = torch.mean(torch.stack([cy_forward, cy_backward]), dim=0)
        output = torch.mean(torch.stack([torch.stack(output_inner_forward), torch.stack(output_inner_backward)]), dim=0)

        return output, (hy, cy), new_mask