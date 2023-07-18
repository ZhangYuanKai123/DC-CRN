# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size=(1,4),stride=2, use="conv"):
        super(GatedConv, self).__init__()
        if use == "conv":
            self.gated_conv = nn.Conv2d(in_channels, out_channles, kernel_size=kernel_size, stride=stride,padding=(0,1))
        elif use == "deconv":
            self.gated_conv = nn.ConvTranspose2d(in_channels, out_channles, kernel_size=kernel_size, stride=stride,padding=(0,1))
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x.timefrequency_dim = x.shape[2]
        half = int(x.timefrequency_dim / 2)
        x_left = x[:, :, :half, :]
        x_right = x[:, :, half:, :]
        x_left = self.gated_conv(x_left)
        x_right = self.gated_conv(x_right)
        x_right = self.activate(x_right)
        out = torch.mul(x_left, x_right)
        return out



class DC_Block(nn.Module):
    def __init__(self, in_channels, out_channels, use="conv",bridge=False):
        super(DC_Block, self).__init__()
        self.conv_dc_block = []
        num = 5
        mid_channel = in_channels
        for block in range(num):
            if block!=num-1:
                self.conv_dc_block.append(nn.Conv2d(mid_channel,out_channels=8,kernel_size=(1,3),padding=(0,1)))
            else:
                if not bridge:
                    self.conv_dc_block.append(GatedConv(mid_channel,out_channels,use=use))
                else:
                    self.conv_dc_block.append(GatedConv(mid_channel, out_channels,kernel_size=(1,3),stride=1, use=use))
            mid_channel += 8

    def forward(self, x):
        out = x
        layerout = 0
        for layer in self.conv_dc_block:
            layerout = layer(out)
            if isinstance(layer, GatedConv):
                break
            out = torch.cat([out, layerout], dim=1)
        return layerout



class DC_CRN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DC_CRN, self).__init__()
        self.dcblock_encoder = []
        self.dcblock_decoder = []
        self.dcblock_bridge = []
        encoder_out_channels_list = [16,32,64,128,256]
        decoder_out_channels_list = [256,128,64,32,16]
        temp = 5
        for i in range(5):
            if i==0:
                self.dcblock_bridge.append(DC_Block(encoder_out_channels_list[0],encoder_out_channels_list[i],bridge=True))
                self.dcblock_encoder.append(DC_Block(in_channels,encoder_out_channels_list[i]))
                self.dcblock_decoder.append(DC_Block(temp,decoder_out_channels_list[i],use="deconv"))
            else:
                self.dcblock_bridge.append(DC_Block(encoder_out_channels_list[i-1],encoder_out_channels_list[i],bridge=True))
                self.dcblock_encoder.append(DC_Block(encoder_out_channels_list[i-1],encoder_out_channels_list[i]))
                self.dcblock_decoder.append(DC_Block(decoder_out_channels_list[i-1], decoder_out_channels_list[i], use="deconv"))
        self.bilstm = nn.LSTM(input, hidden_size=512, num_layers=4, bidirectional=True,batch_first=True)
        self.linear = nn.Linear(161,161)
    def forward(self,x):
        reverse_list = []
        for block1,block2 in zip(self.dcblock_decoder,self.dcblock_bridge):
            x = block1(x)
            reverse_list.append(block2(x))
        tempa = 1024
        tempd = 512
        x = x.view((tempa,tempd))
        x = self.bilstm(x)
        for brige_res,encoder in zip(reversed(reverse_list),self.dcblock_encoder):
            x = torch.cat((x,brige_res),dim=2)
            x = encoder(x)
        split_x = torch.split(x,split_size_or_sections=2,dim=2)
        split_x_left = self.linear(split_x[0])
        split_x_right = self.linear(split_x[1])
        out = torch.cat((split_x_left,split_x_right),dim=2)
        return out



