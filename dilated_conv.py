from keras.layers import Conv1D, Lambda
import keras.backend as K
from SetLearningRate import SetLearningRate


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = SetLearningRate(Conv1D(dim * 2, 3, padding='same', dilation_rate=dilation_rate), lamb=10, is_ada=True)(seq)

    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h

    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq
