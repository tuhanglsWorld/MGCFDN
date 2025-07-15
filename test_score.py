import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.MGCFDN import MultiGranularityConsistencyForgeryDetectionNet
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


def calculate_pixel_f1_all_score(pd, gt):
	if pd.shape != gt.shape:
		print("pred must be same shape with gt")
		return
	pd, gt = pd.flatten(), gt.flatten()
	seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
	true_pos = float(np.logical_and(pd, gt).sum())
	true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
	false_pos = np.logical_and(pd, gt_inv).sum()
	false_neg = np.logical_and(seg_inv, gt).sum()
	return true_pos, true_neg, false_pos, false_neg


class CommonDataset(torch.utils.data.Dataset):
	'''
		load dataset
	'''
	
	def __init__(self, data_txt_dir, transform=None, target_transform=None):
		super(CommonDataset, self).__init__()
		self.transform = transform
		self.target_transform = target_transform
		target_img_file = []
		mask_img_file = []
		mask_target_img_file = []
		three_mask_img_file = []
		with open(data_txt_dir) as f:
			for line in f.readlines():
				img_file, mask_file, target_mask_file, three_mask_file = line.split(',')
				target_img_file.append(img_file.rstrip("\n"))
				mask_img_file.append(mask_file.rstrip("\n"))
				mask_target_img_file.append(target_mask_file.rstrip("\n"))
				three_mask_img_file.append(three_mask_file.rstrip("\n"))
		self.target_img_file = target_img_file
		self.mask_img_file = mask_img_file
		self.mask_target_img_file = mask_target_img_file
		self.three_mask_img_file = three_mask_img_file
	
	def __getitem__(self, index: int):
		target_img_file_image = self.load_image_RGB(self.target_img_file[index], self.transform)
		mask_img_file_image = self.load_image(self.mask_img_file[index], self.target_transform)
		mask_target_img_file_image = self.load_image(self.mask_target_img_file[index], self.target_transform)
		three_mask_img_file_image = self.load_image_RGB(self.three_mask_img_file[index], self.target_transform)
		return target_img_file_image, mask_img_file_image, mask_target_img_file_image, three_mask_img_file_image
	
	def __len__(self) -> int:
		return len(self.target_img_file)
	
	def load_image_RGB(self, image_path, transform=None):
		return transform(Image.open(image_path).convert("RGB"))
	
	def load_image(self, image_path, transform=None):
		return transform(Image.open(image_path).convert("L"))


if __name__ == "__main__":
	model = MultiGranularityConsistencyForgeryDetectionNet()
	data_transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
	])
	mask_transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor(),
	])
	
	test_set = CommonDataset('./dataset/coverage_three_data/all.txt', data_transform, mask_transform)
	test_params = {'batch_size': 1,
	               'shuffle': False,
	               'drop_last': True,
	               'num_workers': 1}
	test_generator = DataLoader(test_set, **test_params)
	
	model.load_state_dict(torch.load('./weight/MGCFDN.pth'))
	model.eval()
	TRUE_POS, TRUE_NEG, FALSE_POS, FALSE_NEG = [], [], [], []
	progress_bar = tqdm(test_generator)
	for iter, data in enumerate(progress_bar):
		imgs, gts, target_gts, st_gts = data
		mask = gts[0].permute(1, 2, 0)
		mask = np.array(mask).astype(np.uint8) * 255
		mask_st = st_gts[0].permute(1, 2, 0)
		mask_st = np.array(mask_st).astype(np.uint8) * 255
		with torch.no_grad():
			lo_mask = model(imgs)
			
			lo_mask = torch.sigmoid(lo_mask)
			lo_mask[lo_mask < 0.5] = 0
			lo_mask[lo_mask >= 0.5] = 1
			lo_mask = lo_mask.permute(0, 2, 3, 1).numpy()[0]
			lo_mask = np.array(lo_mask).astype(np.uint8) * 255
			
			# plt.subplot(131)
			# plt.imshow(imgs[0].permute(1, 2, 0))
			# plt.title(label='original')
			#
			# plt.subplot(132)
			# plt.imshow(mask)
			# plt.title(label='mask')
			#
			# plt.subplot(133)
			# plt.imshow(lo_mask)
			# plt.title(label='lo_mask')
			#
			# plt.show()
			
			true_pos, true_neg, false_pos, false_neg = calculate_pixel_f1_all_score(lo_mask, mask)
			
			TRUE_POS.append(true_pos)
			TRUE_NEG.append(true_neg)
			FALSE_POS.append(false_pos)
			FALSE_NEG.append(false_neg)
			
			true_pos_mean, true_neg_mean, false_pos_mean, false_neg_mean = np.sum(TRUE_POS), np.sum(TRUE_NEG), np.sum(
				FALSE_POS), np.sum(FALSE_NEG)
			precision_real = true_pos_mean / (true_pos_mean + false_pos_mean + 1e-6)
			recall_real = true_pos_mean / (true_pos_mean + false_neg_mean + 1e-6)
			f1_real = (2 * precision_real * recall_real) / (precision_real + recall_real + 1e-6)
			acc_real = (true_pos_mean + true_neg_mean) / (
					true_pos_mean + true_neg_mean + false_pos_mean + false_neg_mean + 1e-6)
			progress_bar.set_description(
				'Test Iteration: {}/{}. F1-score: {:.5f}. Precision: {:.5f}. Recall: {:.5f}. Acc: {:.5f}.'.format(
					iter + 1, len(test_generator), f1_real, precision_real, recall_real, acc_real))
	
	true_pos_mean, true_neg_mean, false_pos_mean, false_neg_mean = np.sum(TRUE_POS), np.sum(TRUE_NEG), np.sum(
		FALSE_POS), np.sum(FALSE_NEG)
	precision = true_pos_mean / (true_pos_mean + false_pos_mean + 1e-6)
	recall = true_pos_mean / (true_pos_mean + false_neg_mean + 1e-6)
	f1 = (2 * precision * recall) / (precision + recall + 1e-6)
	acc = (true_pos_mean + true_neg_mean) / (true_pos_mean + true_neg_mean + false_pos_mean + false_neg_mean + 1e-6)
	print('F1-score: {:.5f}. Precision: {:.5f}. Recall: {:.5f}. Acc: {:.5f}.'.format(f1, precision, recall, acc))




