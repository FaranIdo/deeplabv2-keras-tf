#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from utils import palette, calculate_accuracy

from deeplabv2_model import DeeplabV2


# predicts an image, with the cropping policy of deeplab (single scale for simplicity)
def predict(img, model, crop_size):
    img = img.astype(np.float32)
    h, w, c = img.shape
    c_h, c_w = crop_size

    assert (c_h >= 500 and c_w >= 500), 'Crop size should be greater than 500 for VOC12.'

    pad_height = max(c_h - h, 0)
    pad_width = max(c_w - w, 0)

    # Expand the size of the images to match pad_height, padding right-bottom
    x = cv2.copyMakeBorder(src=img, top=0, bottom=pad_height, left=0, right=pad_width,
                           borderType=cv2.BORDER_CONSTANT, value=np.array([104.008, 116.669, 122.675]))

    # Subtract the values of the padded border so border will be zero
    x[:, :, 0] -= 104.008
    x[:, :, 1] -= 116.669
    x[:, :, 2] -= 122.675

    x_batch = np.expand_dims(x, axis=0)  # Convert to 4d tensor where first dim is batch count=1
    prob = model.predict(x_batch)[0]  # remove batch dimension
    prob = prob[0:h:, 0:w, :]  # resize to match original image
    pred = np.argmax(prob, axis=2)  # get maximum along channels (labels option)
    return pred


def visualize(img, pred):
    # convert prediction to color
    pred_image = palette[pred.ravel()].reshape(img.shape)
    # visualize results
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    a.set_title('Image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(pred_image)
    a.set_title('Segmentation')
    plt.show(fig)


def main():
    model = DeeplabV2(input_shape=(512, 512, 3), apply_softmax=False)
    model.summary()

    # predict image
    # Note - imread open the image in BGR order
    img = cv2.imread('imgs_deeplabv2/2007_000129.jpg')
    pred = predict(img=img, model=model, crop_size=(512, 512))

    gt_img = Image.open('imgs_deeplabv2/GT/2007_000129.png')
    label_image = np.array(gt_img)
    calculate_accuracy(label_image, pred)
    visualize(img, pred)


if __name__ == '__main__':
    main()
