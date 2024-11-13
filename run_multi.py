import numpy as np
from Model.model_multi import DiseasesPredictor
import time
import pickle


file_path = "./data/sample_data/sample_garph"
node_list = pickle.load(open(file_path + ".nodes.pkl", "rb"))
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
rare_patient = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))

# print("node_list:", len(node_list))
# print("adj_lists:", len(adj_lists.keys()))
# print("rare_patient:", len(rare_patient))
# print("labels:", len(labels))
# print("node_map:", len(node_map.keys()))
# print("train:", len(train))
# print("test:", len(test))
# print(rare_patient)
# exit()

# print(train)

# print("sum in labels[len(labels)-1]:", sum(labels[len(labels)-1]))
# Get labels that have more than 1 disease
# multi_labels = []
# for i in range(len(labels)):
#     if sum(labels[i]) > 1:
#         multi_labels.append(i)

# print("multi_labels:", multi_labels)

multi_class_num = 108
feature_dim = 10000
epoch = 8000
batch_num = 200
lr = 0.3
feat_data = np.random.random((50000, feature_dim))
train_enc_dim = [1000, 1000, 1000, 1000]
t1 = time.time()
model = DiseasesPredictor(feat_data=feat_data,
                          b_labels=rare_patient,
                          multi_class_num=108,
                          labels=labels,
                          adj_lists=adj_lists,
                          feature_dim=feature_dim,
                          train_enc_num=1,
                          train_enc_dim=train_enc_dim,
                          train_sample_num=[5, 5, 5, 5],
                          train=train, test=test,
                          kernel='GCN',
                          topk=(1, 2, 3, 4, 5,))

model.run(epoch, batch_num, lr)  # epoch, batch_num, lr
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
