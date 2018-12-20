# 2018.12.20 by mercurial
# Tend to converge to a local optimal solution
import numpy as np

initial_w = 0
initial_b = 0
initial_mu = 1

def parameter_init(shape):
	w = np.ones((shape, ))
	w = w * initial_w
	b = initial_b
	mu = initial_mu
	return w, b, mu

def data_read(filename):
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

def error_count(data, labels, w, b):
	count = 0
	length = labels.shape[0]
	for index in range(length):
		temp = np.dot(w, data[index]) + b
		if labels[index] * temp <= 0:
			count += 1
	return count

def model_train(data, labels, w, b, mu):
	length = labels.shape[0]
	w_test = np.zeros((data.shape[1], ))
	b_test = 0
	while True:
		for index in range(length):
			if labels[index] * (np.dot(w, data[index]) + b) <= 0:
				w_test = w + mu * labels[index] * data[index]
				b_test = b + mu * labels[index]
				print(w_test)
				print(b_test)
				if error_count(data, labels, w_test, b_test) < error_count(data, labels, w, b):
					w = w_test
					b = b_test
					break
		else:
			break
	return w, b

if __name__ == "__main__":
	train_data, train_labels = data_read("traindata.txt")
	w, b, mu = parameter_init(train_data.shape[1])
	w, b = model_train(train_data, train_labels, w, b, mu)
	print(w)
	print(b)		
