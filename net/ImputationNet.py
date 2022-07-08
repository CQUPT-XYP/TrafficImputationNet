from torch import nn
import torch.nn.functional as F
import torch

class ImputationNet(nn.Module):
    def __init__(self, subnet1, subnet2):
        super(ImputationNet, self).__init__()
        self.subnet1 = subnet1
        self.subnet2 = subnet2
        self.merge_layer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, block, center, mask):
        subnet1_output = self.subnet1(block, mask)  # ([15, 9, 1, 128, 128]) ([15, 1, 128, 128])
        # print(block.shape, block.squeeze(2).shape, block[:,5,...].unsqueeze(1).shape, mask.shape)
        subnet2_output, _ = self.subnet2(block.squeeze(2), torch.cat([mask for _ in range(block.shape[1])], dim=1))
        # subnet2_output, _ = self.subnet2(block[:,5,...], mask)
        stack = torch.cat([subnet1_output, subnet2_output], dim=1)
        # print(stack.shape)
        merge_result = self.merge_layer(stack)
        return merge_result
        # return subnet2_output