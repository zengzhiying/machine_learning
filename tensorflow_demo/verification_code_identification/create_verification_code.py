#!/usr/bin/env python
# coding=utf-8
# 生成验证码图片
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
import uuid

# 获取随机验证码字符
def get_random_char():
    i = random.randint(1, 3)
    if i == 1:
        # 输出大写字母 不包括O
        t = chr(random.randint(65, 90))
        while t == 'O':
            t = chr(random.randint(65, 90))
        return t
    elif i == 2:
        return chr(random.randint(97, 122))
    else:
        return str(random.randint(0, 9))

# 随机背景颜色
def get_random_bgcolor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机字体颜色
def get_random_ftcolor():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

# 保存一张验证码图片
def save_verify_code_image(height, witdh, image_type='jpeg'):
    # 创建纯白画布
    image = Image.new('RGB', (witdh, height), (255, 255, 255))
    # 创建font对象
    font = ImageFont.truetype('arial.ttf', 36) 
    # 创建draw绘图对象
    draw = ImageDraw.Draw(image)
    # 填充背景颜色 行遍历
    # for x in range(witdh):
    #     for y in range(height):
    #         draw.point((x, y), fill=get_random_bgcolor())
    # 绘制文字
    v_code = []
    for t in range(4):
        c = get_random_char()
        draw.text((45*t + 12, 10), c, font=font, fill=get_random_ftcolor())
        v_code.append(c)

    # 模糊处理
    # image = image.filter(ImageFilter.BLUR)

    line_num = random.randint(1, 3)
    for i in range(line_num):
        draw.line((random.randint(1, 68), random.randint(1, 68), random.randint(1, 180), random.randint(1, 180)), 
                    fill=get_random_bgcolor())

    image.save('./images/%s_%s.jpg' % (str(uuid.uuid1()).replace('-', ''), ''.join(v_code)), image_type)



if __name__ == '__main__':
    # 图片宽高
    height = 60
    witdh = 60 * 3

    # 生成100000个样本
    for i in range(100000):
        save_verify_code_image(height, witdh)
        if i % 10 == 0:
            print "生成验证码样本: %d个" % i

