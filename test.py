import numpy as np
import torch
import cv2
from model.MGCFDN import MultiGranularityConsistencyForgeryDetectionNet
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image




if __name__ == "__main__":
    model = MultiGranularityConsistencyForgeryDetectionNet()
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = data_transform(Image.open('./image/84t.tif').convert("RGB")).unsqueeze(dim=0).repeat(2, 1, 1, 1)

    model.load_state_dict(torch.load('./weight/MGCFDN.pth'))
    model.eval()
    with torch.no_grad():
        pre_result = model(img)
        pre_result = torch.sigmoid(pre_result)
        pre_result[pre_result < 0.5] = 0
        pre_result[pre_result >= 0.5] = 1
        pre_result = pre_result.permute(0, 2, 3, 1).numpy()[0]
        pre_result = np.array(pre_result).astype(np.uint8) * 255
        plt.subplot(121)
        plt.imshow(img[0].permute(1, 2, 0))
        plt.title(label='original')

        plt.subplot(122)
        plt.imshow(pre_result,'gray')
        plt.title(label='pre_result')
        plt.show()
        cv2.imwrite('./image/result.png', pre_result)
