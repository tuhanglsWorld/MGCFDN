import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.MGCFDN import MultiGranularityConsistencyForgeryDetectionNet
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from prettytable import PrettyTable

def calculate_pixel_f1_all_score(pd, gt):
    if pd.shape != gt.shape:
        print("pred must be same shape with gt")
        return
    if pd.shape[2] == 1:
        pd, gt = pd.flatten(), gt.flatten()
        seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
        true_pos = float(np.logical_and(pd, gt).sum())
        true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
        false_pos = np.logical_and(pd, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, gt).sum()
        return true_pos, true_neg, false_pos, false_neg
    else:
        bg_true_pos, bg_true_neg, bg_false_pos, bg_false_neg = 0, 0, 0, 0
        source_true_pos, source_true_neg, source_false_pos, source_false_neg = 0, 0, 0, 0
        target_true_pos, target_true_neg, target_false_pos, target_false_neg = 0, 0, 0, 0
        for channel in range(3):
            pds, gts = pd[:, :, channel].flatten(), gt[:, :, channel].flatten()
            seg_inv, gt_inv = np.logical_not(pds), np.logical_not(gts)
            true_pos = float(np.logical_and(pds, gts).sum())
            true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
            false_pos = np.logical_and(pds, gt_inv).sum()
            false_neg = np.logical_and(seg_inv, gts).sum()

            if channel == 0:
                target_true_pos, target_true_neg, target_false_pos, target_false_neg = true_pos, true_neg, false_pos, false_neg
            if channel == 1:
                source_true_pos, source_true_neg, source_false_pos, source_false_neg = true_pos, true_neg, false_pos, false_neg
            if channel == 2:
                bg_true_pos, bg_true_neg, bg_false_pos, bg_false_neg = true_pos, true_neg, false_pos, false_neg
        return [bg_true_pos, bg_true_neg, bg_false_pos, bg_false_neg], [source_true_pos, source_true_neg,
                                                                        source_false_pos, source_false_neg], [
            target_true_pos, target_true_neg, target_false_pos, target_false_neg]


def tablePrint(bg, source, target):
    bg_true_pos, bg_true_neg, bg_false_pos, bg_false_neg = bg
    source_true_pos, source_true_neg, source_false_pos, source_false_neg = source
    target_true_pos, target_true_neg, target_false_pos, target_false_neg = target
    # 背景
    bg_true_pos_mean, bg_true_neg_mean, bg_false_pos_mean, bg_false_neg_mean = np.sum(bg_true_pos), np.sum(
        bg_true_neg), np.sum(bg_false_pos), np.sum(bg_false_neg)
    bg_precision = bg_true_pos_mean / (bg_true_pos_mean + bg_false_pos_mean + 1e-6)
    bg_recall = bg_true_pos_mean / (bg_true_pos_mean + bg_false_neg_mean + 1e-6)
    bg_f1 = (2 * bg_precision * bg_recall) / (bg_precision + bg_recall + 1e-6)
    # 源
    source_true_pos_mean, source_true_neg_mean, source_false_pos_mean, source_false_neg_mean = np.sum(
        source_true_pos), np.sum(source_true_neg), np.sum(source_false_pos), np.sum(source_false_neg)
    source_precision = source_true_pos_mean / (source_true_pos_mean + source_false_pos_mean + 1e-6)
    source_recall = source_true_pos_mean / (source_true_pos_mean + source_false_neg_mean + 1e-6)
    source_f1 = (2 * source_precision * source_recall) / (source_precision + source_recall + 1e-6)
    # 目标
    target_true_pos_mean, target_true_neg_mean, target_false_pos_mean, target_false_neg_mean = np.sum(
        target_true_pos), np.sum(target_true_neg), np.sum(target_false_pos), np.sum(target_false_neg)
    target_precision = target_true_pos_mean / (target_true_pos_mean + target_false_pos_mean + 1e-6)
    target_recall = target_true_pos_mean / (target_true_pos_mean + target_false_neg_mean + 1e-6)
    target_f1 = (2 * target_precision * target_recall) / (target_precision + target_recall + 1e-6)

    true_pos_mean = bg_true_pos_mean + source_true_pos_mean + target_true_pos_mean
    true_neg_mean = bg_true_neg_mean + source_true_neg_mean + target_true_neg_mean
    false_pos_mean = bg_false_pos_mean + source_false_pos_mean + target_false_pos_mean
    false_neg_mean = bg_false_neg_mean + source_false_neg_mean + target_false_neg_mean
    acc = (true_pos_mean + true_neg_mean) / (
            true_pos_mean + true_neg_mean + false_pos_mean + false_neg_mean + 1e-6)
    table = PrettyTable(['-', 'F1', 'Precision', 'Recall'])
    table.add_row(['bg', round(bg_f1 * 100, 2), round(bg_precision * 100, 2), round(bg_recall * 100, 2)])
    table.add_row(
        ['source', round(source_f1 * 100, 2), round(source_precision * 100, 2), round(source_recall * 100, 2)])
    table.add_row(
        ['target', round(target_f1 * 100, 2), round(target_precision * 100, 2), round(target_recall * 100, 2)])

    prtstr = "acc: {:.4f} ,bg_f1: {:.4f} ,bg_precision: {:.4f},bg_recall: {:.4f}/" \
             "source_f1: {:.4f} ,source_precision: {:.4f},source_recall: {:.4f}/" \
             "target_f1: {:.4f} ,target_precision: {:.4f},target_recall: {:.4f}"
    prtstr = prtstr.format(acc, bg_f1, bg_precision, bg_recall, source_f1, source_precision, source_recall, target_f1,
                           target_precision, target_recall)
    return table, acc, bg_f1, source_f1, target_f1, prtstr


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
            for line in f.readlines():  ##readlines(),函数把所有的行都读取进来；
                img_file, mask_file, target_mask_file, three_mask_file = line.split(',')  ##删除行后的换行符，img_file 就是每行的内容啦
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

    model = MultiGranularityConsistencyForgeryDetectionNet(out_channel=3)
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

    model.load_state_dict(torch.load('./weight/MGCFDN_ST.pth'))
    model.eval()
    TRUE_POS, TRUE_NEG, FALSE_POS, FALSE_NEG = [], [], [], []

    # 进度条
    progress_bar = tqdm(test_generator)
    BG_TRUE_POS, BG_TRUE_NEG, BG_FALSE_POS, BG_FALSE_NEG = [], [], [], []
    SOURCE_TRUE_POS, SOURCE_TRUE_NEG, SOURCE_FALSE_POS, SOURCE_FALSE_NEG = [], [], [], []
    TARGET_TRUE_POS, TARGET_TRUE_NEG, TARGET_FALSE_POS, TARGET_FALSE_NEG = [], [], [], []
    softmax = nn.Softmax2d()
    for iter, data in enumerate(progress_bar):
        imgs, gts, target_gts, st_gts = data
        mask = gts[0].permute(1, 2, 0)
        mask = np.array(mask).astype(np.uint8) * 255
        mask_st = st_gts[0].permute(1, 2, 0)
        mask_st = np.array(mask_st).astype(np.uint8) * 255
        with torch.no_grad():
            lo_mask = model(imgs)
            lo_mask = softmax(lo_mask)
            lo_mask[lo_mask < 0.5] = 0
            lo_mask[lo_mask >= 0.5] = 1
            lo_mask = lo_mask.permute(0, 2, 3, 1).numpy()[0]
            lo_mask = np.array(lo_mask).astype(np.uint8) * 255

            [bg_true_pos, bg_true_neg, bg_false_pos, bg_false_neg], [source_true_pos, source_true_neg, source_false_pos,
                                                                     source_false_neg], [target_true_pos,
                                                                                         target_true_neg,
                                                                                         target_false_pos,
                                                                                         target_false_neg] = calculate_pixel_f1_all_score(
                lo_mask, mask_st)

            BG_TRUE_POS.append(bg_true_pos)
            BG_TRUE_NEG.append(bg_true_neg)
            BG_FALSE_POS.append(bg_false_pos)
            BG_FALSE_NEG.append(bg_false_neg)
            SOURCE_TRUE_POS.append(source_true_pos)
            SOURCE_TRUE_NEG.append(source_true_neg)
            SOURCE_FALSE_POS.append(source_false_pos)
            SOURCE_FALSE_NEG.append(source_false_neg)
            TARGET_TRUE_POS.append(target_true_pos)
            TARGET_TRUE_NEG.append(target_true_neg)
            TARGET_FALSE_POS.append(target_false_pos)
            TARGET_FALSE_NEG.append(target_false_neg)
        table, acc, bg_f1, source_f1, target_f1, prtstr = tablePrint(
            [BG_TRUE_POS, BG_TRUE_NEG, BG_FALSE_POS, BG_FALSE_NEG],
            [SOURCE_TRUE_POS, SOURCE_TRUE_NEG, SOURCE_FALSE_POS, SOURCE_FALSE_NEG],
            [TARGET_TRUE_POS, TARGET_TRUE_NEG, TARGET_FALSE_POS, TARGET_FALSE_NEG])
        progress_bar.set_description(
            'Test Iteration: {}/{}. 【s/t】{}'.format(iter + 1, len(test_generator), prtstr))

    table, acc, bg_f1, source_f1, target_f1, prtstr = tablePrint(
        [BG_TRUE_POS, BG_TRUE_NEG, BG_FALSE_POS, BG_FALSE_NEG],
        [SOURCE_TRUE_POS, SOURCE_TRUE_NEG, SOURCE_FALSE_POS, SOURCE_FALSE_NEG],
        [TARGET_TRUE_POS, TARGET_TRUE_NEG, TARGET_FALSE_POS, TARGET_FALSE_NEG])
    print('\n【s/t】Test. Acc: {:.4f} \n {}'.format(acc, table))
