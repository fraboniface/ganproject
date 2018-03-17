import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self, input_dim, n_feature_maps):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			#1x1
			nn.ConvTranspose2d(input_dim, 8*n_feature_maps, 4, 1, 0, bias=False),
			nn.BatchNorm2d(8*n_feature_maps),
			nn.ReLU(True),
			#4x4
			nn.ConvTranspose2d(8*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(4*n_feature_maps),
			nn.ReLU(True),
			#8x8
			nn.ConvTranspose2d(4*n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(2*n_feature_maps),
			nn.ReLU(True),
			#16x16
			nn.ConvTranspose2d(2*n_feature_maps, n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(n_feature_maps),
			nn.ReLU(True),
			#32x32
			nn.ConvTranspose2d(n_feature_maps, 3, 4, 2, 1, bias=False),
			#64x64
			nn.Tanh()
		)

	def  forward(self, x):
		return self.main(x)


class ACDiscriminator(nn.Module):
	def __init__(self, n_class1, n_class2, n_feature_maps):
		super(ACDiscriminator, self).__init__()
		self.n_class1 = n_class1
		self.n_class2 = n_class2

		self.main = nn.Sequential(
			#64x64
			nn.Conv2d(3, n_feature_maps, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			#32x32
			nn.Conv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(2*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			#16x16
			nn.Conv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(4*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			#8x8
			nn.Conv2d(4*n_feature_maps, 8*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(8*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			# 4x4
			nn.Conv2d(8*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False)
			#1x1
		)
		self.source = nn.Sequential(
			nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
			nn.Sigmoid()
		)
		self.class1_logits = nn.Conv2d(n_feature_maps, self.n_class1, 1, 1, 0, bias=False)
		self.class2_logits = nn.Conv2d(n_feature_maps, self.n_class2, 1, 1, 0, bias=False)

	def  forward(self, x, fm=False, only_fm=False):
		x = self.main(x)
		if only_fm:
			return x

		source = self.source(x).view(-1, 1).squeeze(1)
		class1_logits = self.class1_logits(x).view(-1, self.n_class1)
		class2_logits = self.class2_logits(x).view(-1, self.n_class2)

		if fm:
			return x, class1_logits, class2_logits
		else:
			return source, class1_logits, class2_logits


class Q_ACDiscriminator(nn.Module):
	def __init__(self, n_class1, n_class2, n_info_vars, n_feature_maps):
		super(Q_ACDiscriminator, self).__init__()
		self.n_class1 = n_class1
		self.n_class2 = n_class2
		self.n_info_vars = n_info_vars

		self.main = nn.Sequential(
			#64x64
			nn.Conv2d(3, n_feature_maps, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			#32x32
			nn.Conv2d(n_feature_maps, 2*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(2*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			#16x16
			nn.Conv2d(2*n_feature_maps, 4*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(4*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			#8x8
			nn.Conv2d(4*n_feature_maps, 8*n_feature_maps, 4, 2, 1, bias=False),
			nn.BatchNorm2d(8*n_feature_maps),
			nn.LeakyReLU(0.2, inplace=True),
			# 4x4
			nn.Conv2d(8*n_feature_maps, n_feature_maps, 4, 1, 0, bias=False)
			#1x1
		)
		self.source = nn.Sequential(
			nn.Conv2d(n_feature_maps, 1, 1, 1, 0, bias=False),
			nn.Sigmoid()
		)
		self.class1_logits = nn.Conv2d(n_feature_maps, self.n_class1, 1, 1, 0, bias=False)
		self.class2_logits = nn.Conv2d(n_feature_maps, self.n_class2, 1, 1, 0, bias=False)
		self.info_params = nn.Conv2d(n_feature_maps, 2*self.n_info_vars, 1, 1, 0, bias=False)


	def  forward(self, x, mode='D', input_source='fake', only_fm=False):
		assert mode in ['D', 'G']
		assert input_source in ['real', 'fake']
		x = self.main(x)
		if only_fm:
			return x

		class1_logits = self.class1_logits(x).view(-1, self.n_class1)
		class2_logits = self.class2_logits(x).view(-1, self.n_class2)

		if mode == 'D':
			source = self.source(x).view(-1, 1).squeeze(1)
			if input_source == 'real':
				return source, class1_logits, class2_logits
			else:
				Qc_x = self.info_params(x).view(-1, 2*self.n_info_vars)
				mean = Qc_x[-1,:self.n_info_vars]
				var = torch.exp(Qc_x[-1,self.n_info_vars:])
				return source, class1_logits, class2_logits, mean, var
		else:
			Qc_x = self.info_params(x).view(-1, 2*self.n_info_vars)
			mean = Qc_x[-1,:self.n_info_vars]
			var = torch.exp(Qc_x[-1,self.n_info_vars:])
			return x, class1_logits, class2_logits, mean, var