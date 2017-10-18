#!/usr/bin/env python
# coding=utf-8
"""使用keras以及ImageNet训练集实现简单的图片中物体识别
来源: https://github.com/DeepLearningSandbox/DeepLearningSandbox
"""
import sys
import argparse
from PIL import Image

import numpy as np
#import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
target_size = (224, 224)

def predict(model, img, target_size, top_n=3):
    """Run model prediction on image
    Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
    Returns:
    list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=top_n)[0]

def plot_preds(image, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
    """
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    order = list(reversed(range(len(preds))))
    bar_preds = [pr[2] for pr in preds]
    labels = (pr[1] for pr in preds)
    plt.barh(order, bar_preds, alpha=0.5)
    plt.yticks(order, labels)
    plt.xlabel('Probability')
    plt.xlim(0,1.01)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # 编译命令行参数
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--image", help="path to image")
    args = args_parser.parse_args()
    # print(args)

    if args.image is None:
        args_parser.print_help()
        sys.exit(1)

    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    #plot_preds(img, preds)
    print(preds)
