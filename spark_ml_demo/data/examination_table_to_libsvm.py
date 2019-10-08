#!/usr/bin/env python
# coding=utf-8

if __name__ == '__main__':
    with open('table.txt') as fp:
        num = 0
        # 新建要保存的文件
        f_new = open('data3.txt', 'wb')
        for line in fp:
            if num != 0:
                info = line[:-1].split(',')
                print info
                # 获取身高 体重 龋齿数
                height = info[3]
                body_weight = info[4]
                caries_number = info[5]

                # 获取性别 男:1  女:0
                gender = 0
                if info[1] == '男':
                    gender = 1

                f_new.write('%d 1:%s 2:%s 3:%s' % (gender, height, body_weight, caries_number))
                f_new.write('\n')

            num += 1

        f_new.close()
        print "save ok! number: %d" % (num - 1)
