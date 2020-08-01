import os
import numpy as np
import cv2

import torch
from torchvision import transforms

MODEL_NAME = 'model1'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Provided model are entirely saved. if testing, use following code
    # replace NUM to test model name

    # from pytorch_model1 import Net
    # model = Net().to(DEVICE)
    # model.load_state_dict(torch.load('./NUM.pth', map_location=DEVICE))
    # torch.save(model, './%s.pth' % MODEL_NAME)

    model = torch.load('./%s.pth' % MODEL_NAME, map_location=DEVICE)

    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    for root, dirs, files in os.walk('./input', topdown=False):
        for name in files:
            print(os.path.join(root, name))

            im = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)

            res = model(preprocess(im).unsqueeze(0).to(DEVICE))

            im_res = (res.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()) * 255

            cv2.imwrite(os.path.join('./output', name), im_res)


if __name__ == "__main__":
    main()
