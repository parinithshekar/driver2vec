from matplotlib.style import reload_library
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,padding,dilation_base,dropout):
        super(TCN, self).__init__()
        self.layers = []

        self.num_layers = torch.ceil(torch.log2( ((input_size-1)*(dilation_base-1))/((kernel_size-1)*2) +1 ))
    
        for i in len(self.num_layers):
            dilation = dilation_base ** i
            if i == 0:
                layers += [ResidualBlock(input_size,
                                     output_size,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation,
                                     padding=(kernel_size - 1) * dilation,
                                     dropout=dropout)]
            else:
                layers += [ResidualBlock(output_size,
                                     output_size,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation,
                                     padding=(kernel_size - 1) * dilation,
                                     dropout=dropout)]

        self.TCNnet = nn.Sequential(*layers)

    def forward(self, x):
        out = self.TCNnet(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,padding,dilation,dropout):
        super(ResidualBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.conv1 = weight_norm(CausalConvLayer(input_size,output_size,kernel_size,padding,dilation))
        self.conv2 = weight_norm(CausalConvLayer(output_size,output_size,kernel_size,padding,dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.resblock = nn.Sequential(
            self.conv1,
            self.relu,
            self.dropout,
            self.conv2,
            self.relu,
            self.dropout
        )

        self.residual_conv = nn.Conv1d(input_size, output_size, 1)
        #Initialize weights and bias
        nn.init.kaiming_uniform_(self.resicual_conv.weight)
        nn.init.kaiming_uniform_(self.resicual_conv.bias)

    def forward(self, x):
        if self.input_size == self.output_size:
            out = self.resblock(x) + self.residual_conv(x)
        else:
            out = self.resblock(x)
        return out


class CausalConvLayer(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,padding,dilation):
        super(CausalConvLayer, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(input_size,output_size,kernel_size,stride=1,padding=padding,dilation=dilation)
        #Initialize weights and bias
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.kaiming_uniform_(self.conv.bias)


    def forward(self, x):
        out = self.conv(x)
        out = out[:,:,:-self.padding]       #for causal convulution, only the left padding is used, the right padding needs to be removed
        return out

