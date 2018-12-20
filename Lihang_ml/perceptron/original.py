# 2018.12.19 by mercurial
import numpy as np

initial_w = 0
initial_b = 0
initial_mu = 1

def data_list(data, labels, line):
	line = line.splitlines()[0]
	str_list = line.split(", ")
	int_list = []
	for var in str_list[:-1]:
		int_list.append(int(var))
	data.append(int_list)
	labels.append(int(str_list[-1]))

def read_datasets(filename):
	data = []
	labels = []
	with open(filename, "r") as filepointer:
		for line in filepointer:
			data_list(data, labels, line)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

def initialize(shape):
	w = np.ones((shape, ))
	w = initial_w * w
	b = initial_b
	mu = initial_mu
	return w, b, mu

def find_error(data, labels, w, b):
	length = labels.shape[0]
	for index in range(length):
		judge_num = labels[index] * (np.vdot(w, data[index]) + b)
		if judge_num <= 0:
			return index
	return -1

def model_train(data, labels, w, b, mu):
	index = find_error(data, labels, w, b)
	while index >= 0:
		w = w + mu * labels[index] * data[index]
		b = b + mu * labels[index]
		index = find_error(data, labels, w, b)
	return w, b 	

def test_testsets(data, labels, w, b):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	length = labels.shape[0]
	for index in range(length):
		prediction = np.vdot(w, data[index]) + b
		if prediction > 0:
			if labels[index] > 0:
				tp += 1
			else:
				fp += 1
		else:
			if labels[index] > 0:
				fn += 1
			else:
				tn += 1
	accuracy = (tp + tn) / length
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * precision * recall / ( precision + recall)
	return accuracy, precision, recall, f1

def model_predict(filename, w, b):
	data_predict = []
	with open(filename, "r") as filepointer:
		for line in filepointer:
			line = line.splitlines()[0]
			str_list = line.split(", ")
			int_list = []
			for var in str_list:
				int_list.append(int(var))
			data_predict.append(int_list)
	predictions = []
	length = data_predict.shape[0]
	for index in range(length):
		cal_predict = np.vdot(w, data_predict[index]) + b
		if cal_predict > 0:
			predictions.append(1)
		else:
			predictions.append(-1)
	predictions = np.array(predictions)
	return predictions

if __name__ == "__main__":
	train_data, train_labels = read_datasets("traindata.txt")
	test_data, test_labels = read_datasets("testdata.txt")
	
	w, b, mu = initialize(train_data.shape[1])
	w, b = model_train(train_data, train_labels, w, b, mu)

	accuracy, precision, recall, f1 = test_testsets(test_data, test_labels, w, b)
	print("the accuracy is: ", accuracy)
	print("the precision is: ", precision)
	print("the recall is: ", recall)
	print("the f1 is: ", f1)
