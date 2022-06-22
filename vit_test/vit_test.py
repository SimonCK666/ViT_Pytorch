'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-06-21 12:29:33
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-06-21 19:10:24
FilePath: /vit_test/vit_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    # num_classes = 1000,
    num_classes = 1024,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

# sequence: [1024, 128, 96] ==> [131072, 96]

preds = v(img)      # (1, num_classes)
# print(preds.shape)