import keras.backend as K
from keras.layers import Dense, Lambda, Dropout
import numpy as np
import math
from keras_layer_normalization import LayerNormalization
from SetLearningRate import SetLearningRate


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def CrossAttention(x):
    # 针对之前的实验结果 怀疑下游层的网络学习率过低导致难以收敛 用lstm处理nyt数据集的时候好像出现这种问题
    # 目前用lstm训练nyt数据集的时候将lr调到0.001->0.0001
    # 基于这种情况 保证pre-train bert的lr为0.00001->0.000005的前提下
    # 对decoder层的网络lr调整为0.0001->0.00005
    # 效果未知 预计下次实验的时候进行调整
    sub_features, tokens_feature = x
    tokens_feature = Dropout(0.5)(tokens_feature)
    cross_key_layer = SetLearningRate(Dense(512, activation="relu", kernel_initializer="glorot_normal"), lamb=10, is_ada=True)(sub_features)
    cross_value_layer = SetLearningRate(Dense(512, activation="relu", kernel_initializer="glorot_normal"), lamb=10, is_ada=True)(sub_features)
    self_query_layer = SetLearningRate(Dense(512, activation="relu", kernel_initializer="glorot_normal"), lamb=10, is_ada=True)(tokens_feature)
    attention_score = Lambda(lambda x: K.batch_dot(x[0], K.permute_dimensions(x[1], (0, 2, 1))) / np.sqrt(64))([self_query_layer, cross_key_layer])
    attention_score = Lambda(lambda x: K.softmax(attention_score, -1))(attention_score)
    attention_drop = Dropout(0.4)(attention_score)
    context_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attention_drop, cross_value_layer])
    context_layer = SetLearningRate(Dense(1024, activation="relu"),lamb=10,is_ada=True)(context_layer)
    context_layer = SetLearningRate(Dense(768, activation=None),lamb=10,is_ada=True)(context_layer)
    context_layer = Lambda(gelu)(context_layer)
    context_layer_Drop = Dropout(0.5)(context_layer)
    context_layer_LN = LayerNormalization()(context_layer_Drop)
    tokens_feature = Lambda(lambda x: x[0] + x[1])([tokens_feature, context_layer_LN])
    return tokens_feature


def Muilt_head(x):
    # 事实证明 多头注意力机制在这里没用
    # 不仅没用 反而有可能反作用
    # ps：现在怀疑是不是和学习率有一定层度的关系
    # 采用pre-train bert进行fine tune时需要考虑搭配网络的lr问题
    cross_key_layer, cross_value_layer, self_query_layer = x
    cross_key_layer = K.reshape(cross_key_layer, (-1, K.shape(cross_key_layer)[1], 12, 64))
    cross_value_layer = K.reshape(cross_value_layer, (-1, K.shape(cross_value_layer)[1], 12, 64))
    self_query_layer = K.reshape(self_query_layer, (-1, K.shape(self_query_layer)[1], 12, 64))
    cross_key_layer = K.permute_dimensions(cross_key_layer, (0, 2, 1, 3))
    cross_value_layer = K.permute_dimensions(cross_value_layer, (0, 2, 1, 3))
    self_query_layer = K.permute_dimensions(self_query_layer, (0, 2, 1, 3))
    return cross_key_layer, cross_value_layer, self_query_layer


def recover(x):
    x = K.permute_dimensions(x, (0, 2, 1, 3))
    x = K.reshape(x, (-1, K.shape(x)[1], 768))
    return x

# 目前的实验似乎证明在这里多头注意力机制好像没什么卵用
# 而且crossattention的mask机制好像也没什么卵用
# Transformer里面采用mask是防止信息泄露
# 但在这里似乎并没用什么作用，如果想mask的话似乎sotamax完直接mask也比原来性能好
# 反正就挺迷的
