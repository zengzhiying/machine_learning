#!/usr/bin/env python
# coding=utf-8
# 分割验证码图片为小图片
from PIL import Image
import uuid
import os

def split_verify_code_image(file_name):
    image = Image.open(file_name)
    text = file_name[:-4].split('_')[1]
    box1 = (0, 0, 45, 60)
    box2 = (45, 0, 90, 60)
    box3 = (90, 0, 135, 60)
    box4 = (135, 0, 180, 60)
    image.crop(box1).save('train_images/%s_%s.jpg' % (str(uuid.uuid1()).replace('-',''), text[0]))
    image.crop(box2).save('train_images/%s_%s.jpg' % (str(uuid.uuid1()).replace('-',''), text[1]))
    image.crop(box3).save('train_images/%s_%s.jpg' % (str(uuid.uuid1()).replace('-',''), text[2]))
    image.crop(box4).save('train_images/%s_%s.jpg' % (str(uuid.uuid1()).replace('-',''), text[3]))


if __name__ == '__main__':
    image_files = []
    for parent, dirnames, filenames in os.walk('./images'):
        for image_file in filenames:
            split_verify_code_image('images/%s' % image_file)
            image_files.append(image_file)
            print "分割张数: %d" % len(image_files)
