# 2018.12.20 by mercurial
import numpy as np

mu = 1

def data_fromfile(filename):
	data = []
	labels = []
	with open(filename, "r") as filepointer:
		for line in filepointer:
			line = line.splitlines()[0]
			str_list = line.split(", ")
			int_list = []
			for var in str_list[:-1]:
				int_list.append(int(var))
			data.append(int_list)
			labels.append(int(str_list[-1]))
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

def model_train(data, labels):
	gram_matrix = np.dot(data, data.T)
	co_matrix = np.zeros(data.shape[0])
	length = data.shape[0]
	while True:
		for index in range(length):
			temp = co_matrix * labels
			temp = np.vdot(temp, gram_matrix[index]) + np.vdot(co_matrix, labels)
			if temp * labels[index] <= 0:
				co_matrix[index] += mu
				break
		else:
			break
	return co_matrix

def cal_wb(co_matrix, data, labels):
	temp = co_matrix * labels
	w = np.dot(temp, data)
	b = np.sum(temp)
	return w, b

if __name__ == "__main__":
	train_data, train_labels = data_fromfile("traindata.txt")
	test_data, test_labels = data_fromfile("testdata.txt")
	
	co_matrix = model_train(train_data, train_labels)	
	#print(co_matrix)
	w, b = cal_wb(co_matrix, train_data, train_labels)
	print(w)
	print(b)
