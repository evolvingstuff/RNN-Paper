import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random

SANITY_CHECK = False

def pathologicalMultiplyGenerator(sequence_length):
	global SANITY_CHECK
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
	if SANITY_CHECK == False:
		Y = [product]
	else:
		Y = [random.random()*random.random()]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalMultiplication'

def pathologicalAddingGenerator(sequence_length):
	global SANITY_CHECK
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
	if SANITY_CHECK == False:
		Y = [tot]
	else:
		Y = [random.random() + random.random()]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalAdding'

def pathologicalXorGenerator(sequence_length):
	global SANITY_CHECK
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
	if SANITY_CHECK == False:
		Y = [float(tot%2)]
	else:
		Y = [float(random.randint(0,1))]
	return torch.tensor(X), torch.tensor(Y), 'pathologicalXOR'

def sequentialParitySequenceGenerator(sequence_length):
	global SANITY_CHECK
	tot = 0
	X = []
	for t in range(sequence_length):
		if random.random() < 0.5:
			X.append([0.0])
		else:
			X.append([1.0])
			tot += 1
	if SANITY_CHECK == False:
		Y = [float(tot%2)]
	else:
		Y = [float(random.randint(0,1))]
	return torch.tensor(X), torch.tensor(Y), 'sequentialParity'

##################################################################

def batchGenerator(sequence_length, batch_size, sequence_generator):
	bX, bY = [], []
	for b in range(batch_size):
		X, Y, task_name = sequence_generator(sequence_length)
		bX.append(X)
		bY.append(Y)
	return torch.stack(bX,1), torch.stack(bY,0), task_name

##################################################################

class AbsDiagCell(torch.nn.Module):

	def __init__(self, input_size, hidden_size):
		super(AbsDiagCell, self).__init__()
		self.IH = nn.Linear(input_size, hidden_size, bias=False)
		self.HH = nn.Parameter(torch.ones((hidden_size)))

	def forward(self, x, h):
		return torch.abs(self.IH(x)+self.HH*h)

	def clamp(self):
		self.HH.data = torch.clamp(self.HH.data, -1.0, 1.0)

class AbsDiagNet(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(AbsDiagNet, self).__init__()
		self.hidden_size = hidden_size
		self.recurrent_layer = AbsDiagCell(input_size, hidden_size)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		h = torch.zeros(batch_size, self.hidden_size)
		for step in X:
			h = self.recurrent_layer(step, h)
		Y = self.HO(h)
		return Y

	def clamp(self):
		self.recurrent_layer.clamp()

class AbsDiagNetGated(torch.nn.Module):

	def __init__(self, input_size, gate_size, hidden_size, output_size):
		super(AbsDiagNetGated, self).__init__()
		self.hidden_size = hidden_size
		self.relu = nn.ReLU()
		self.IH = nn.Linear(input_size, gate_size, bias=False) #TODO: yes or no?
		self.recurrent_layer = AbsDiagCell(gate_size, hidden_size)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)
		
	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		h = torch.zeros(batch_size, self.hidden_size)
		for step in X:
			g = self.relu(self.IH(step))
			h = self.recurrent_layer(g, h)
		Y = self.HO(h)
		return Y

	def clamp(self):
		self.recurrent_layer.clamp()

class LSTMNet(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(LSTMNet, self).__init__()
		self.hidden_size = hidden_size
		self.recurrent_layer = nn.LSTMCell(input_size, hidden_size)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		h, c = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
		for step in X:
			h, c = self.recurrent_layer(step, (h, c))
		Y = self.HO(h)
		return Y

##################################################################

if __name__ == '__main__':

	print('')
	print('')
	print('-----------------------------------')
	print('RNN demo')

	SANITY_CHECK = False
	seed = 132 #128
	if seed != None:
		random.seed(seed)
		torch.manual_seed(seed)
		print('seed='+str(seed))

	use_abs_diag_rnn = True # False -> use lstm

	global_norm_clip = 30.0 #TODO
	local_grad_clip = 1.0

	seq_length = 300 #TODO
	total_training_examples = 1000 #1000
	show_every = 50
	batch_size_train = 20
	batch_size_test = 1000
	gate_size = 20
	hidden_size = 40

	shuffle_training_examples = False

	input_size = 2 #TODO
	output_size = 1
	timesteps = 1000

	lr = 0.001
	momentum = 0
	alpha = 0.999
	weight_decay = 0.001

	if use_abs_diag_rnn:
		#net = AbsDiagNet(input_size,hidden_size,output_size) #TODO
		net = AbsDiagNetGated(input_size, gate_size, hidden_size, output_size)
		print('using Abs Diag Net')
	else:
		net = LSTMNet(input_size, hidden_size, output_size)
		print('using LSTM')

	#sequence_generator = sequentialParitySequenceGenerator #TODO
	#sequence_generator = pathologicalXorGenerator
	sequence_generator = pathologicalAddingGenerator
	#sequence_generator = pathologicalMultiplyGenerator
	
	testX, testY, task_name = batchGenerator(seq_length, batch_size_test, sequence_generator)

	print('task: ' +task_name+', length = ' + str(seq_length))
	print('')

	optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, alpha=alpha, weight_decay=weight_decay)
	#optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001, weight_decay=0.9995)

	loss_fn = nn.MSELoss()
	#loss_fn = nn.BCEWithLogitsLoss()
	#loss_fn = nn.L1Loss()

	training_examples = []
	for example in range(total_training_examples):
		trainX, trainY, _ = batchGenerator(seq_length, batch_size_train, sequence_generator)
		training_examples.append((trainX, trainY))


	for t in range(timesteps):

		if shuffle_training_examples:
			random.shuffle(training_examples)

		s = 0
		for example in training_examples:

			trainX, trainY = example[0], example[1]
			predictedY = net(trainX)
			loss = loss_fn(predictedY, trainY)

			optimizer.zero_grad()

			loss.backward()
			
			#Important to apply both kinds of gradient clipping
			torch.nn.utils.clip_grad_norm_(net.parameters(), global_norm_clip)
			torch.nn.utils.clip_grad_value_(net.parameters(), local_grad_clip)

			optimizer.step()

			if use_abs_diag_rnn:
				net.clamp()

			if s % show_every == 0:
				predictedY = net(testX)
				loss = loss_fn(predictedY, testY)
				print('batch {:d} of {:d}\ttest loss: {:f}'.format(t+1, timesteps, loss.item()))

			s += 1