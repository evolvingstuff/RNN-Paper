import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random

def sequentialParitySequence(sequence_length):
	tot = 0
	X = []
	for t in range(sequence_length):
		if random.random() < 0.5:
			X.append([0.0])
		else:
			X.append([1.0])
			tot += 1
	Y = [float(tot%2)]
	return torch.tensor(X), torch.tensor(Y)

def sequentialParityBatch(sequence_length, batch_size):
	bX = []
	bY = []
	for b in range(batch_size):
		X, Y = sequentialParitySequence(sequence_length)
		bX.append(X)
		bY.append(Y)
	return torch.stack(bX,1), torch.stack(bY,0)

class AbsDiagRNNCell(torch.nn.Module):

	def __init__(self, input_size, hidden_size):
		super(AbsDiagRNNCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.IH = nn.Linear(input_size, hidden_size, bias=False)
		self.HH = nn.Parameter(torch.ones((hidden_size)))

	def forward(self, x0, h0):
		input_batch_size, input_size = x0.size()
		hidden_batch_size, hidden_size = h0.size()
		assert input_batch_size == hidden_batch_size
		assert input_size == self.input_size
		assert hidden_size == self.hidden_size
		h1 = torch.abs(self.IH(x0)+self.HH*h0)
		return h1

class AbsDiagNet(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(AbsDiagNet, self).__init__()
		self.hidden_size = hidden_size
		self.rnn = AbsDiagRNNCell(input_size, hidden_size)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		#print(X.size())

		h = torch.zeros(batch_size, self.hidden_size)
		for step in X:
			h = self.rnn(step, h)
		Y = self.HO(h)
		return Y

	def clamp(self):
		self.rnn.HH.data = torch.clamp(self.rnn.HH.data, -1.0, 1.0)

class LSTMNet(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size):
		super(LSTMNet, self).__init__()
		self.hidden_size = hidden_size
		self.rnn = nn.LSTMCell(input_size, hidden_size)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		h, c = torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)
		for step in X:
			h, c = self.rnn(step, (h, c))
		Y = self.HO(h)
		return Y

if __name__ == '__main__':

	print('')
	print('')
	print('-----------------------------------')
	print('AbsDiagRNNCell demo')

	use_abs_diag_rnn = True # False -> use lstm

	seq_length = 1000
	batch_size = 100
	hidden_size = 10

	if use_abs_diag_rnn:
		net = AbsDiagNet(1,hidden_size,1)
		print('Using Abs Diag Net')
	else:
		net = LSTMNet(1, hidden_size, 1)
		print('Using LSTM')

	print('Sequential parity, length = ' + str(seq_length))

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.RMSprop(net.parameters())
	#optimizer = torch.optim.Adam(net.parameters())

	print('')
	print('')

	for t in range(5000):
		bX, bY = sequentialParityBatch(seq_length, batch_size)
		pred_bY = net(bX)
		loss = loss_fn(pred_bY, bY)
		print(t, loss.item())
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
		optimizer.step()
		if use_abs_diag_rnn:
			net.clamp()