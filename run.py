#! -*- coding:utf-8 -*-


from Model import model, Evaluate

maxlen = 100
learning_rate = 1e-4
from utils import get_tokenizer
from data_loader import data_generator, load_data
from EMA import ExponentialMovingAverage

config_path = 'uncased_L-12_H-768_A-12/bert_config.json'  # bert pre_trained weight
checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

train_data_path = 'Data/WebNLG/train_triples.json'  # data_path
dev_data_path = 'Data/WebNLG/dev_triples.json'
test_data_path = 'Data/WebNLG/test_triples.json'
rel_dict_path = 'Data/WebNLG/rel2id.json'

tokenizer = get_tokenizer(dict_path)  # 构造字典
train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_data_path, dev_data_path, test_data_path, rel_dict_path)  # load_data
train_generator = data_generator(train_data, tokenizer, rel2id, num_rels, maxlen=maxlen, batch_size=5)  # 生成迭代器
subject_model, object_model, hbt_model = model(config_path, checkpoint_path, learning_rate, num_rels)  # 构造模型
EMAer = ExponentialMovingAverage(hbt_model)
EMAer.inject()
steps = len(train_data) // 5

evaluator = Evaluate(subject_model, object_model, tokenizer, id2rel, dev_data, 'best_model.weights', ema=EMAer)  # 定义回调机制
hbt_model.fit_generator(train_generator.__iter__(),  # train
                        steps_per_epoch=steps,
                        epochs=100,
                        callbacks=[evaluator],
                        verbose=2
                        )
