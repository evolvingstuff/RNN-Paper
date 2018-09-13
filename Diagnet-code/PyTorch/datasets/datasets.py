import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random

def pathologicalMultiplyGenerator(sequence_length):
	t1 = random.randint(0, sequence_length/2-1)
	t2 = random.randint(sequence_length/2,sequence_length-1)
	assert t1 < t2
	assert t2 < sequence_length
	product = 1.0
	X = []
	for t in range(sequence_length):
		x1 = random.random()
		if t == t1 or t == t2:
			x2 = 1.0
			product *= x1
		else:
			x2 = 0.0
		X.append([x1, x2])
	Y = [product]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalMultiplication'

def pathologicalAddingGenerator(sequence_length):
	t1 = random.randint(0, sequence_length/2-1)
	t2 = random.randint(sequence_length/2,sequence_length-1)
	assert t1 < t2
	assert t2 < sequence_length
	tot = 0.0
	X = []
	for t in range(sequence_length):
		x1 = random.random()
		if t == t1 or t == t2:
			x2 = 1.0
			tot += x1
		else:
			x2 = 0.0
		X.append([x1, x2])
	Y = [tot]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalAdding'

def pathologicalXorGenerator(sequence_length):
	t1 = random.randint(0, sequence_length/2-1)
	t2 = random.randint(sequence_length/2,sequence_length-1)
	assert t1 < t2
	assert t2 < sequence_length
	tot = 0
	X = []
	for t in range(sequence_length):
		if random.random() < 0.5:
			x1 = 0.0
		else:
			x1 = 1.0
		if t == t1 or t == t2:
			x2 = 1.0
			if x1 > 0:
				tot += 1
		else:
			x2 = 0.0
		X.append([x1, x2])
	Y = [float(tot%2)]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalXOR'

def sequentialParitySequenceGenerator(sequence_length):
	tot = 0
	X = []
	for t in range(sequence_length):
		if random.random() < 0.5:
			X.append([0.0])
		else:
			X.append([1.0])
			tot += 1
	Y = [float(tot%2)]
	return torch.tensor(X), torch.tensor(Y), 'sequentialParity'

def gradCheckSequenceGenerator(sequence_length):
	tot = 0
	X = []
	for t in range(sequence_length):
		X.append([random.random()])
	Y = [random.random()]
	return torch.tensor(X, requires_grad=True), torch.tensor(Y, requires_grad=True), 'gradCheck'

##################################################################

def batchGenerator(sequence_length, batch_size, sequence_generator):
	bX, bY = [], []
	for b in range(batch_size):
		X, Y, task_name = sequence_generator(sequence_length)
		bX.append(X)
		bY.append(Y)
	return torch.stack(bX,1), torch.stack(bY,0), task_name


##################################################################