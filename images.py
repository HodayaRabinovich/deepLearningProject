# showing images of the test and after applying noise

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import PIL

colorJitter_transforms = {
        'jitter': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        ]),
    }


# show noisy images
im = cv2.imread("./data/test/AFRICAN CROWNED CRANE/2.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis('off')
plt.imshow(im)
plt.show()

noiseR = np.random.randn(im.shape[0], im.shape[1])*0.1
noiseG = np.random.randn(im.shape[0], im.shape[1])*0.1
noiseB = np.random.randn(im.shape[0], im.shape[1])*0.1
im[:, :, 0] = im[:, :, 0] + noiseR
im[:, :, 1] = im[:, :, 1] + noiseG
im[:, :, 2] = im[:, :, 2] + noiseB
noisy_im = im
plt.figure()
plt.axis('off')
plt.imshow(noisy_im)
plt.show()

PILim = TF.to_pil_image(im)
CJim = TF.adjust_brightness(PILim, 0.6)
CJim = TF.adjust_contrast(CJim, 0.4)
CJim = TF.adjust_saturation(CJim, 0.4)
CJim = TF.adjust_hue(CJim, 0.2)
plt.figure()
plt.axis('off')
plt.imshow(CJim)
plt.show()
