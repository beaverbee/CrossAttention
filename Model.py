from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.regularizers import l2
from keras_bert import load_trained_model_from_checkpoint
from utils import get_k_inter, metric
from CrossAttention import CrossAttention
from dilated_conv import dilated_gated_conv1d
import numpy as np


def model(bert_config_path, bert_checkpoint_path, learning_rate, num_rels):
    bert_model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path, seq_len=None)  # 载入预训练模型
    for l in bert_model.layers:
        l.trainable = True

    tokens_in = Input(shape=(None,))  # [batch_size,len]
    segments_in = Input(shape=(None,))  # [batch_size,len]
    gold_sub_heads_in = Input(shape=(None,))  # [batch_size,len,1]  用0，1标注是否为实体
    gold_sub_tails_in = Input(shape=(None,))
    sub_head_in = Input(shape=(1,))  # 和苏神的标注方式一致 此处应该是输入位置信息 [batch_size,1]
    sub_tail_in = Input(shape=(1,))
    gold_obj_heads_in = Input(shape=(None, num_rels))  # 此处应该是 [batch_size,len,relation_classes]
    gold_obj_tails_in = Input(shape=(None, num_rels))

    tokens, segments, gold_sub_heads, gold_sub_tails, sub_head, sub_tail, gold_obj_heads, gold_obj_tails = tokens_in, segments_in, gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(tokens)  # 生成mask矩阵  mask=[batch_size, len, 1]维度一致

    tokens_feature = bert_model([tokens, segments])  # token feature   [batch_size, len(sentence_max_length), hidden_dim(768)]
    tokens_feature = Lambda(lambda x: x, output_shape=lambda s: s)(tokens_feature)
    tokens_feature = Dropout(0.5)(tokens_feature)
    tokens_feature = dilated_gated_conv1d(tokens_feature, mask, dilation_rate=3)
    pred_sub_heads = Dense(1, activation='sigmoid')(tokens_feature)  # predict subject start   [batch_size, len, 1]
    pred_sub_tails = Dense(1, activation='sigmoid')(tokens_feature)  # predict subject end     [batch_size, len, 1]

    subject_model = Model([tokens_in, segments_in], [pred_sub_heads, pred_sub_tails])  # 构造subject model
    t_dim = K.int_shape(tokens_feature)[-1]
    sub_features = Lambda(get_k_inter, output_shape=(6, t_dim))([tokens_feature, sub_head, sub_tail])
    tokens_feature_drop=Dropout(0.5)(tokens_feature)
    Cross_attention_feature = CrossAttention([sub_features, tokens_feature_drop])
    pred_obj_heads = Dense(num_rels, activation='sigmoid')(Cross_attention_feature)  # predict object_relation start
    pred_obj_tails = Dense(num_rels, activation='sigmoid')(Cross_attention_feature)  # predict object_relation end

    object_model = Model([tokens_in, segments_in, sub_head_in, sub_tail_in], [pred_obj_heads, pred_obj_tails])  # 构造object_relation model

    hbt_model = Model([tokens_in, segments_in, gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in],
                      [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])  # 构造train_model

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2)
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2)

    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)  # loss function
    sub_heads_loss = K.sum(sub_heads_loss * mask) / K.sum(mask)
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * mask) / K.sum(mask)

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), 2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), 2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * mask) / K.sum(mask)

    loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

    hbt_model.add_loss(loss)
    hbt_model.compile(optimizer=Adam(learning_rate))  # optimizer setting
    hbt_model.summary()

    return subject_model, object_model, hbt_model


class Evaluate(Callback):
    def __init__(self, subject_model, object_model, tokenizer, id2rel, eval_data, save_weights_path, min_delta=1e-4, patience=10, ema=None):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.subject_model = subject_model
        self.object_model = object_model
        self.tokenizer = tokenizer
        self.id2rel = id2rel
        self.eval_data = eval_data
        self.save_weights_path = save_weights_path
        self.ema = ema
        self.stage = 0

    def on_train_begin(self, logs=None):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.warmup_epochs = 2
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.ema.apply_ema_weights()
        precision, recall, f1 = metric(self.subject_model, self.object_model, self.eval_data, self.id2rel, self.tokenizer)
        if self.monitor_op(f1 - self.min_delta, self.best) or self.monitor_op(self.min_delta, f1):  # 获得最大F1值保存模型
            self.best = f1
            self.wait = 0
            self.model.save_weights(self.save_weights_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:  # 连续7周期F1值没有提高停止训练
                self.stopped_epoch = epoch
                self.model.stop_training = True
        self.ema.reset_old_weights()
        if (
                epoch > 15 and (f1 < 0.5 or self.wait > 10) and self.stage == 0
        ):
            self.stage = 1
            self.model.load_weights('best_model.weights')
            self.ema.initialize()
            K.set_value(self.model.optimizer.lr, 5e-6)
            K.set_value(self.model.optimizer.iterations, 0)
            opt_weights = K.batch_get_value(self.model.optimizer.weights)
            opt_weights = [w * 0. for w in opt_weights]
            K.batch_set_value(zip(self.model.optimizer.weights, opt_weights))
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
