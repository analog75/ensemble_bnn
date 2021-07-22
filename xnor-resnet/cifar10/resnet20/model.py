import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SimpleBinaryActivation(nn.Module):
#{{{
    def __init__(self):
        super(SimpleBinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out
#}}}

class HardBinaryConv(nn.Module):
#{{{
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y
#}}}

class BasicBlock_XNOR(nn.Module):
#{{{
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_XNOR, self).__init__()
        self.binary_activation1 = SimpleBinaryActivation()
        self.binary_activation2 = SimpleBinaryActivation()
        self.binaryconv1 = HardBinaryConv(in_planes, planes, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.binaryconv2 = HardBinaryConv(planes, planes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.prelu1=nn.PReLU();
        self.prelu2=nn.PReLU();
        self.prelu3=nn.PReLU();

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = self.binary_activation1(out)
        out = self.prelu1(self.binaryconv1(out))
        out = self.bn2(out)
        out = self.binary_activation2(out)
        out = self.binaryconv2(out)
        out = self.prelu2(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.prelu3(out)
        return out
#}}}

class ResNet_XNOR(nn.Module):
#{{{
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_XNOR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.dropout = nn.Dropout()
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out
#}}}

def ResNet20_XNOR():
    return ResNet_XNOR(BasicBlock_XNOR, [3,3,3])

def ResNet32_XNOR():
    return ResNet_XNOR(BasicBlock_XNOR, [5,5,5])
