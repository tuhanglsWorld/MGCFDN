import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer
from timm.models.layers import DropPath
import copy



class FFN(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
			nn.BatchNorm2d(hidden_features),
		)
		self.act = act_layer(act)
		self.fc2 = nn.Sequential(
			nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
			nn.BatchNorm2d(out_features),
		)
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		x = self.drop_path(x) + shortcut
		return x


class Stem(nn.Module):
	""" Image to Visual Word Embedding
	Overlap: https://arxiv.org/pdf/2106.13797.pdf
	"""
	
	def __init__(self, img_size=256, in_dim=3, out_dim=768, act='relu'):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
			nn.BatchNorm2d(out_dim // 8),
			act_layer(act),
			nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
			nn.BatchNorm2d(out_dim // 4),
			act_layer(act),
			nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
			nn.BatchNorm2d(out_dim // 2),
			act_layer(act),
			nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
			nn.BatchNorm2d(out_dim),
			act_layer(act),
			nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
			nn.BatchNorm2d(out_dim),
		)
	
	def forward(self, x):
		x = self.convs(x)
		return x


class DeepGCN(torch.nn.Module):
	def __init__(self, drop_path_rate=0.1, drop_rate=0.1, num_knn=9):
		super(DeepGCN, self).__init__()
		
		self.k = num_knn  # neighbor num (default:9)
		self.conv = 'mr'  # graph conv layer {edge, mr}
		self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
		self.norm = 'batch'  # batch or instance normalization {batch, instance}
		self.bias = True  # bias of conv layer True or False
		self.n_blocks = 16  # number of basic blocks in the backbone
		self.channels = 640  # number of channels of deep features
		self.dropout = drop_rate  # dropout rate
		self.use_dilation = True  # use dilated knn or not
		self.epsilon = 0.2  # stochastic epsilon for gcn
		self.stochastic = False  # stochastic for gcn, True or False
		self.drop_path = drop_path_rate
		
		self.stem = Stem(out_dim=self.channels, act=self.act)
		dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.n_blocks)]  # stochastic depth decay rule
		num_knn = [int(x.item()) for x in torch.linspace(self.k, 2 * self.k, self.n_blocks)]  # number of knn's k
		max_dilation = 196 // max(num_knn)
		
		self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, 16, 16))
		
		if self.use_dilation:
			self.backbone = Seq(
				*[Seq(Grapher(self.channels, num_knn[i], min(i // 4 + 1, max_dilation), self.conv, self.act, self.norm,
				              self.bias, self.stochastic, self.epsilon, 1, drop_path=dpr[i]),
				      FFN(self.channels, self.channels * 4, act=self.act, drop_path=dpr[i])
				      ) for i in range(self.n_blocks)])
		else:
			self.backbone = Seq(*[Seq(Grapher(self.channels, num_knn[i], 1, self.conv, self.act, self.norm,
			                                  self.bias, self.stochastic, self.epsilon, 1, drop_path=dpr[i]),
			                          FFN(self.channels, self.channels * 4, act=self.act, drop_path=dpr[i])
			                          ) for i in range(self.n_blocks)])
		#self.model_init()
	
	def model_init(self):
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)
				m.weight.requires_grad = True
				if m.bias is not None:
					m.bias.data.zero_()
					m.bias.requires_grad = True
		print("[loading pre_train]")
		pretrained_dict = torch.load('./vig_b_82.6.pth')
		model_dict = self.state_dict()
		full_dict = copy.deepcopy(pretrained_dict)
		for k in list(full_dict.keys()):
			if k in model_dict:
				if full_dict[k].shape != model_dict[k].shape:
					del full_dict[k]
		
		self.load_state_dict(full_dict, strict=False)
	
	def forward(self, inputs, out_size=(40, 40)):
		x = self.stem(inputs) + self.pos_embed
	
		for i in range(self.n_blocks):
			x = self.backbone[i](x)
		return x





if __name__ == '__main__':
	model = DeepGCN()
	result = torch.randn(size=(8, 3, 256, 256))
	pre_result  = model(result)
	print(pre_result.size())
	total = sum([param.nelement() for param in model.parameters()])
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(num_params)
	total = sum([param.nelement() for param in model.parameters()])
	print("Number of parameter: %.2fM" % (total / 1e6))