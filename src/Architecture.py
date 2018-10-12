import os, pickle, numpy as np


train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]


# function provided in the official website
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_train_batch(i):
	data = unpickle(os.path.join("cifar-10-batches-py", train_files[i]))
	x = data[b'data'].reshape(10000, 3, 32, 32)
	y = np.array(data[b'labels']).reshape(10000, 1)
	return x, y


def 