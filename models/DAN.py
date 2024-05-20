import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']
from models.common import *

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        m_batchsize, C, height, width = x.size()

        #(2,2048,16,8)->(2,256,16,8)->(2,256,128)->(2,128,256)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)#(2,256,128)
        energy = torch.bmm(proj_query, proj_key)#(2,128,128)
        attention = self.softmax(energy)#(2,128,128)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)#torch.Size([2, 2048, 128])
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))#torch.Size([2, 2048, 128])
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)#torch.Size([2, 2048, 128])
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)#torch.Size([2, 128, 2048])
        energy = torch.bmm(proj_query, proj_key)#(2,2048,2048)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)#torch.Size([2, 2048, 2048])

        proj_value = x.view(m_batchsize, C, -1)#(2,2048,128)
        out = torch.bmm(attention, proj_value)#torch.Size([2, 2048, 128])
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out
class DAN(Module):
    def __init__(self, in_dim):
        super(DAN, self).__init__()
        self.PAM = PAM_Module(in_dim)
        self.CAM = CAM_Module(in_dim)
        # self.Conv = Conv(in_dim, in_dim, 3)
    def forward(self,x):
        x1 = self.PAM(x)
        x2 = self.CAM(x)
        x = torch.add(x1,x2)
        return x
