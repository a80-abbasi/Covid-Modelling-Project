import torch
from torchvision import transforms
import numpy as np
import cv2
from skimage.filters import rank
from skimage.morphology import disk


class LungSegmentor(object):
    def __init__(self, model, out_size, kernel_size, num_iter, margin, threshold, device):
        self.model = model.to(device)
        self.out_size = out_size
        self.margin = margin
        self.threshold = threshold
        self.device = device
        self.kernel_size = kernel_size
        self.num_iter = num_iter

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.to(self.device)
        probs = self.model(img.unsqueeze(0))[0]
        mask = (probs > self.threshold) * 1.0

        img_cv = np.array(self.to_pil(img))
        mask_cv = mask.cpu().detach().numpy()
        kernel = np.ones((self.kernel_size, self.kernel_size))
        mask_dilated = cv2.dilate(
            mask_cv.squeeze(), kernel=kernel, iterations=self.num_iter)

        mask_y, mask_x = np.where(mask_dilated > 0)
        start_x = max(np.min(mask_x) - self.margin, 0)
        end_x = min(np.max(mask_x) + self.margin, mask_dilated.shape[1])
        start_y = max(np.min(mask_y) - self.margin, 0)
        end_y = min(np.max(mask_y) + self.margin, mask_dilated.shape[0])

        # mask_cropped = mask_dilated[start_y:end_y, start_x:end_x]
        # roi_cropped_resized = cv2.resize(mask_cropped, dsize=self.out_size)

        cropped_img = img_cv.squeeze()[start_y:end_y, start_x:end_x]
        cropped_img_resized = cv2.resize(cropped_img, dsize=self.out_size)

        return self.to_tensor(cropped_img_resized)


class HistEqualizer(object):
    def __init__(self, d=40):
        self.d = d
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        img_cv = np.array(self.to_pil(img))
        img_cv_ahe = np.expand_dims(rank.equalize(
            img_cv.squeeze(), selem=disk(self.d)), axis=-1)
        return self.to_tensor(img_cv_ahe)
