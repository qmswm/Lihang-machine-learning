# 2018.12.20 by mercurial
# Voted-perceptron algorithm
# See details in Large Margin Classification Using the Perceptron Algorithm
import numpy as np

MAX_ITERATE_TIMES = 100
initial_w = 0
initial_b = 0
initial_mu = 1

def parameter_init(shape):
	v = np.ones((shape, ))
	v = v * initial_w
	v[-1] = initial_b
	mu = initial_mu
	return v, mu

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
			int_list.append(1)
			data.append(int_list)
			labels.append(int(str_list[-1]))
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

def model_train(data, labels, v, mu, iterate_times = MAX_ITERATE_TIMES):
	v_iterated = []
	v_iterate_times = []
	length = data.shape[0]
	linearly_separable = False
	while iterate_times > 0:
		for index in range(length):
			if labels[index] * np.vdot(data[index], v) <=0:
				v_iterated.append(v)
				v_iterate_times.append(index)
				v = v + mu * labels[index] * data[index]
				break
		else:
			linearly_separable = True
			break
		iterate_times -= 1
	return v, v_iterated, v_iterate_times, linearly_separable

if __name__ == "__main__":
	train_data, train_labels = data_read("traindata.txt")
	v, mu = parameter_init(train_data.shape[1])
	v, v_iterated, v_iterate_times, linearly_separable = model_train(train_data, train_labels, v, mu)
	if linearly_separable:
		print(v)
	else:
		length = len(v_iterate_times)
		v = np.zeros_like(v)
		for index in range(length):
			v += v_iterate_times[index] * v_iterated[index]
		print(v)		
				
