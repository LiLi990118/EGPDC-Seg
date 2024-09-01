import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):

	def __init__(self, num_classes, ndf):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		x = self.up_sample(x)
		x = self.sigmoid(x)
		return x

if __name__ == '__main__':
    print('#### Test Case ###')
    x = Variable(torch.rand(2,5,224,224)).cuda()
    model = Discriminator(5,64).cuda()
    #param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    #print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    param1 = sum([param.nelement() for param in model.parameters()])
    # param = count_param(model)
    # print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))
    print(param1)
