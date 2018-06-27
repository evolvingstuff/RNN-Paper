import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random

class AbsDiagCell(torch.nn.Module):

	def __init__(self, input_size, hidden_size, init_weight_recur):
		super(AbsDiagCell, self).__init__()
		self.IH = nn.Linear(input_size, hidden_size, bias=False)
		self.HH = nn.Parameter(torch.ones((hidden_size)))
		for i in range(hidden_size):
			self.HH.data[i] = init_weight_recur
		self.debug_mode = False

	def forward(self, x, h):
		return torch.abs(self.IH(x)+self.HH*h)

	def clamp(self):
		self.HH.data = torch.clamp(self.HH.data, -1.0, 1.0)

class AbsDiagNet(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, init_weight_recur=0.99):
		super(AbsDiagNet, self).__init__()
		self.hidden_size = hidden_size
		self.recurrent_layer = AbsDiagCell(input_size, hidden_size, init_weight_recur)
		self.HO = nn.Linear(hidden_size, output_size, bias=True)

	def forward(self, X):
		seq_length, batch_size, input_size = X.size()
		h = torch.zeros(batch_size, self.hidden_size)
		t = 0
		for step in X:
			h = self.recurrent_layer(step, h)
			if self.debug_mode:
				print('['+str(t)+']')
				print('\tinput: ' + str(step))
				print('\thiddn: ' + str(h))
			t += 1
		Y = self.HO(h)
		if self.debug_mode:
			print('\toutpt: ' + str(Y))
		return Y

	def clamp(self):
		self.recurrent_layer.clamp()

class AbsDiagNetGated(torch.nn.Module):

	def __init__(self, input_size, gate_size, hidden_size, output_size, init_weight_recur=0.99):
		super(AbsDiagNetGated, self).__init__()
		self.hidden_size = hidden_size
		self.relu = nn.ReLU()
		self.IH = nn.Linear(input_size, gate_size, bias=False) #TODO: yes or no?
		self.recurrent_layer = AbsDiagCell(gate_size, hidden_size, init_weight_recur)
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