'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-06-17 15:14:35
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-06-17 18:52:39
FilePath: \ViT_Pytorch\vision_transformer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from importlib.resources import path
import torch
import torch.nn as nn
import torch.nn.functional as F


# Step1: convert image to embedding vector sequence
def img2emb_naive(image, patch_size, weight):
    # image shape: batch_size * channel * h * w
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)      # 将图片分块，块之间没有交点，所以stride = patch_size
    patch_embedding = patch @ weight    # @: 矩阵相乘
    return patch_embedding


def img2emb_conv(image, kernel, stride):
    # 先做一个 3d conv: kernel <==> weight
    conv_output = F.conv2d(image, kernel, stride=stride)    # batch_size * output_channel * output_height * output_weight
    batch_size, output_channel, output_height, output_weight = conv_output.shape
    # 将 oh * ow 拉直
    patch_embedding = conv_output.reshape((batch_size, output_channel, output_height * output_weight)).transpose(-1, -2)
    return patch_embedding


# test code for img2emb
batch_size, input_channel, image_h, image_w = 1, 3, 8, 8
# 4 * 4 为一个 patch
patch_size = 4
model_dim = 8
patch_depth = patch_size * patch_size * input_channel
image = torch.randn(batch_size, input_channel, image_h, image_w)

# test img2emb_naive
weight = torch.randn(patch_depth, model_dim)    # model_dim <==> output_channel, patch_depth <==> 卷积核面积 * 输入通道数
# print(weight.shape)     # torch.Size([48, 8])
patch_embedding_naive = img2emb_naive(image, patch_size, weight)    # torch.Size([1, 48, 4]) 48: patch_size * patch_size * input_channel # 1: batch_size # 4: patch number
# print(patch_embedding_naive.shape)      # torch.Size([1, 4, 8]) 一张图片分成四块，一块用8长度的向量表示

# test img2emb_conv
kernel = weight.transpose(0, 1).reshape((-1, input_channel, patch_size, patch_size))       # oc * ic * kh * kw
patch_embedding_conv = img2emb_conv(image, kernel, patch_size)
# print(patch_embedding_conv.shape)


#===========================================================
#===========================================================

# Step2 add classfication token embedding like BERT
# 文章中说这个 embedding 是随机初始化的
cls_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
# 将 patch_embedding 和 cla embedding 拼接
token_embedding = torch.cat([cls_token_embedding, patch_embedding_naive], dim=1)

#===========================================================
#===========================================================

# Step3 add position embedding
max_num_token = 16
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1]
position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
token_embedding += position_embedding


#===========================================================
#===========================================================

# Step4 Send token_embedding into Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
encoder_output = transformer_encoder(token_embedding)

# Step5 do classification
num_classes = 10
label = torch.randint(10, (batch_size, ))
cls_token_output = encoder_output[:, 0, :]
linear_layer = nn.Linear(model_dim, num_classes)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
print(loss)
