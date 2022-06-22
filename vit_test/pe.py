'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-06-21 10:18:40
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-06-21 11:28:30
FilePath: /vit_test/pe.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import numpy as np


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
