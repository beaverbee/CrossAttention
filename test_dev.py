from utils import metric, get_tokenizer
from Model import model, Evaluate
from data_loader import data_generator, load_data

config_path = 'cased_L-12_H-768_A-12/bert_config.json'  # bert pre_trained weight
checkpoint_path = 'cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'cased_L-12_H-768_A-12/vocab.txt'


train_data_path = 'Data/NYT/train_triples.json'  # data_path
dev_data_path = 'Data/NYT/dev_triples.json'
test_data_path = 'Data/NYT/test_triples.json'
rel_dict_path = 'Data/NYT/rel2id.json'

maxlen = 100
learning_rate = 1e-4
tokenizer = get_tokenizer(dict_path)  # 构造字典
train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_data_path, dev_data_path, test_data_path, rel_dict_path)  # load_data
train_generator = data_generator(train_data, tokenizer, rel2id, num_rels, maxlen=maxlen, batch_size=6)  # 生成迭代器
subject_model, object_model, hbt_model = model(config_path, checkpoint_path, learning_rate, num_rels)  # 构造模型
steps = len(train_data) // 6
best_weight_path='best_model_weight/best_model.weights'
hbt_model.load_weights(best_weight_path)
precision, recall, f1 = metric(subject_model, object_model, test_data, id2rel, tokenizer)
print("precision:" + str(precision))
print("recall:" + str(recall))
print("F1:" + str(f1))
