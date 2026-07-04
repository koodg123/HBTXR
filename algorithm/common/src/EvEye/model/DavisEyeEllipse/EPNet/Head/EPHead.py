import torch
import torch.nn as nn
import math
import numpy as np

# HEAD_DICT = {'hm': 1, 'ab': 2, 'ang': 1, 'trig': 2, 'reg': 2, 'mask': 1}
HEAD_DICT = {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}


class EPHead(nn.Module):
    def __init__(self, in_channels, head_conv=256, head_dict=HEAD_DICT):
        super(EPHead, self).__init__()
        self.heads = head_dict
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    head_conv,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
            if head == "hm":
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        outputs = {}
        for head in self.heads:
            outputs[head] = self.__getattr__(head)(x)
        return outputs


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def main():
    def test_head_module():
        # Number of channels in the input feature map
        in_channels = 128
        # Number of channels in the first convolution layer of the output head
        head_conv = 256
        # Create a Head module instance
        head_module = EPHead(in_channels, head_conv, HEAD_DICT)

        # Print the Head module structure
        print(head_module)

        # Create a random input tensor with shape (batch_size, channels, height, width)
        input_tensor = torch.randn(32, in_channels, 64, 64)

        # Forward pass
        output = head_module(input_tensor)

        # Print each output head shape
        for head in output:
            print(f"{head} output shape: {output[head].shape}")

        # # Print the weights of the last convolution layer in each output head
        # for head in HEAD_DICT:
        #     layer = head_module.__getattr__(head)[-1]  # Get the last convolution layer
        #     print(f"{head} head last layer weights: {layer.weight.data}")

    # Run the test function
    test_head_module()


if __name__ == "__main__":
    main()
