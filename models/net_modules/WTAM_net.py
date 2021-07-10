import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *

################################### ***************************  #####################################
###################################         ISCAS-WTAM           #####################################
################################### ***************************  #####################################

class WTAMGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, conv=default_conv, kernel_size=3, act=nn.ReLU(True)):
        super(WTAMGenerator, self).__init__()

        self.deepsupervision = False

        self.DWT = DWT()
        self.IWT = IWT()


        m_head = [BBlock(conv, input_nc, ngf, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(DBlock_com1(conv, ngf, ngf, kernel_size, act=act, bn=False))

        d_l1 = []
        d_l1.append(BBlock(conv, ngf * 4, ngf * 2, kernel_size, act=act, bn=False))
        d_l1.append(DBlock_com1(conv, ngf * 2, ngf * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(BBlock(conv, ngf * 8, ngf * 4, kernel_size, act=act, bn=False))
        d_l2.append(DBlock_com1(conv, ngf * 4, ngf * 4, kernel_size, act=act, bn=False))

        d_l3 = []
        d_l3.append(BBlock(conv, ngf * 16, ngf * 8, kernel_size, act=act, bn=False))
        d_l3.append(DBlock_com1(conv, ngf * 8, ngf * 8, kernel_size, act=act, bn=False))

        pro_l4 = []
        pro_l4.append(BBlock(conv, ngf * 32, ngf * 16, kernel_size, act=act, bn=False))
        pro_l4.append(DBlock_com(conv, ngf * 16, ngf * 16, kernel_size, act=act, bn=False))
        pro_l4.append(DBlock_inv(conv, ngf * 16, ngf * 16, kernel_size, act=act, bn=False))
        pro_l4.append(BBlock(conv, ngf * 16, ngf * 32, kernel_size, act=act, bn=False))

        i_l01 = [DBlock_inv1_att(conv, ngf // 2, ngf // 2, kernel_size, ngf // 2, ngf // 2, ngf // 4, act=act,
                                        bn=False)]
        i_l01.append(BBlock(conv, ngf // 2, ngf, kernel_size, act=act, bn=False))

        i_l11 = [DBlock_inv1_att(conv, ngf, ngf, kernel_size, ngf, ngf, ngf // 2, act=act, bn=False)]
        i_l11.append(BBlock(conv, ngf, ngf * 2, kernel_size, act=act, bn=False))

        i_l21 = [DBlock_inv1_att(conv, ngf * 2, ngf * 2, kernel_size, ngf * 2, ngf * 2, ngf, act=act, bn=False)]
        i_l21.append(BBlock(conv, ngf * 2, ngf * 4, kernel_size, act=act, bn=False))

        i_l3 = [DBlock_inv1_att(conv, ngf * 8, ngf * 8, kernel_size, ngf * 8, ngf * 8, ngf * 4, act=act, bn=False)]
        i_l3.append(BBlock(conv, ngf * 8, ngf * 16, kernel_size, act=act, bn=False))

        i_l02 = [DBlock_inv1_att(conv, ngf // 2, ngf // 2, kernel_size, ngf // 2, ngf // 2, ngf // 4, act=act,
                                        bn=False)]
        i_l02.append(BBlock(conv, ngf // 2, ngf, kernel_size, act=act, bn=False))

        i_l12 = [DBlock_inv1_att(conv, ngf, ngf, kernel_size, ngf, ngf, ngf // 2, act=act, bn=False)]
        i_l12.append(BBlock(conv, ngf, ngf * 2, kernel_size, act=act, bn=False))

        i_l2 = [DBlock_inv1_att(conv, ngf * 4, ngf * 4, kernel_size, ngf * 4, ngf * 4, ngf * 2, act=act, bn=False)]
        i_l2.append(BBlock(conv, ngf * 4, ngf * 8, kernel_size, act=act, bn=False))

        i_l03 = [DBlock_inv1_att(conv, ngf // 2, ngf // 2, kernel_size, ngf // 2, ngf // 2, ngf // 4, act=act,
                                        bn=False)]
        i_l03.append(BBlock(conv, ngf // 2, ngf, kernel_size, act=act, bn=False))

        i_l1 = [DBlock_inv1_att(conv, ngf * 2, ngf * 2, kernel_size, ngf * 2, ngf * 2, ngf, act=act, bn=False)]
        i_l1.append(BBlock(conv, ngf * 2, ngf * 4, kernel_size, act=act, bn=False))

        i_l0 = [DBlock_inv1_att(conv, ngf, ngf, kernel_size, ngf, ngf, ngf // 2, act=act, bn=False)]
        m_tail = [conv(ngf, ngf, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l0 = nn.Sequential(*d_l0)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l3 = nn.Sequential(*d_l3)
        self.pro_l4 = nn.Sequential(*pro_l4)
        self.i_l01 = nn.Sequential(*i_l01)
        self.i_l11 = nn.Sequential(*i_l11)
        self.i_l21 = nn.Sequential(*i_l21)
        self.i_l3 = nn.Sequential(*i_l3)
        self.i_l02 = nn.Sequential(*i_l02)
        self.i_l12 = nn.Sequential(*i_l12)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l03 = nn.Sequential(*i_l03)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(ngf, output_nc, kernel_size=1)
            self.final2 = nn.Conv2d(ngf, output_nc, kernel_size=1)
            self.final3 = nn.Conv2d(ngf, output_nc, kernel_size=1)
            self.final4 = nn.Conv2d(ngf, output_nc, kernel_size=1)
        else:
            self.final = nn.Conv2d(ngf, output_nc, kernel_size=1)

    def forward(self, inputs):

        xd_l0 = self.d_l0(self.head(inputs))
        xd_l1 = self.d_l1(self.DWT(xd_l0))
        xd_l2 = self.d_l2(self.DWT(xd_l1))
        xd_l3 = self.d_l3(self.DWT(xd_l2))
        xpro_l4 = self.pro_l4(self.DWT(xd_l3))

        xi_l01 = self.i_l01(self.IWT(xd_l1)) + xd_l0
        xi_l11 = self.i_l11(self.IWT(xd_l2)) + xd_l1
        xi_l21 = self.i_l21(self.IWT(xd_l3)) + xd_l2
        xi_l3 = self.i_l3((self.IWT(xpro_l4)) + xd_l3)

        xi_l02 = self.i_l02(self.IWT(xi_l11)) + xd_l0 + xi_l01
        xi_l12 = self.i_l12(self.IWT(xi_l21)) + xd_l1 + xi_l11
        xi_l2 = self.i_l2((self.IWT(xi_l3)) + xd_l2 + xi_l21)

        xi_l03 = self.i_l03(self.IWT(xi_l12)) + xd_l0 + xi_l01 + xi_l02
        xi_l1 = self.i_l1((self.IWT(xi_l2)) + xd_l1 + xi_l11 + xi_l12)
        xi_l0 = self.tail(self.i_l0((self.IWT(xi_l1))) + xd_l0 + xi_l01 + xi_l02 + xi_l03)

        if self.deepsupervision:
            output1 = self.final1(xi_l01)
            output2 = self.final2(xi_l02)
            output3 = self.final3(xi_l03)
            output4 = self.final4(xi_l0)
            return [output1, output2, output3, output4]

        else:
            output = self.final(xi_l0)
            return output














