import os

# # Try running on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
from keras.models import load_model

MODEL_NAME = 'model1'
model = load_model('./%s.h5' % MODEL_NAME)

for root, dirs, files in os.walk('./input', topdown=False):
    for name in files:
        print(os.path.join(root, name))

        im = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)

        im_predict = im.reshape((1, im.shape[0], im.shape[1], 1))
        im_predict = im_predict.astype(np.float32) / 255

        result = model.predict(im_predict)

        im_res = result.reshape((result.shape[1], result.shape[2]))
        im_res = im_res * 255

        cv2.imwrite(os.path.join('./output', name), im_res)
