# -*- coding: utf-8 -*-
# @Time : 2023/4/17 7:24 下午
# @Author : chenxiangan
# @File : run.py
# @Software: PyCharm

import image_dataset


file_dict = {}
file_iter = image_dataset.ImageDataset("/Users/chennan/Downloads")
for each_file in file_iter:
    for img_info in each_file:
        file_name = img_info.file_name
        image = img_info.image
        meta = img_info.meta
