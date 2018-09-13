import os

# # Try running on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
from keras.models import load_model

MODEL_NAME = 'model1.h5'
model = load_model(MODEL_NAME)

for root, dirs, files in os.walk('input', topdown=False):
    for name in files:
        print(os.path.join(root, name))

        im = cv2.imread(os.path.join(root, name))
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        im_predict = im_gray.reshape((1, im_gray.shape[0], im_gray.shape[1], 1))
        im_predict = im_predict.astype(np.float32) / 255.

        result = model.predict(im_predict)

        im_res = result.reshape((result.shape[1], result.shape[2]))
        im_res = im_res * 255

        cv2.imwrite(os.path.join('output', name), im_res)
