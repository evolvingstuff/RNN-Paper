import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, random
from models.models import *
from datasets.datasets import *
import javarandom

def randomSequenceMaker(r):
	def generator(sequence_length):
		X = []
		for t in range(sequence_length):
			X.append([r.nextDouble()])
		Y = [float(r.nextDouble())]
		return torch.tensor(X), torch.tensor(Y), 'randomSequence'
	return generator

if __name__ == '__main__':

	print('')
	print('')
	print('-----------------------------------')
	print('RNN test')

	seed = 123
	if seed != None:
		random.seed(seed)
		torch.manual_seed(seed)
		print('seed='+str(seed))

	seed2 = 321

	global_norm_clip = 30.0
	local_grad_clip = 1.0

	seq_length = 10
	total_training_examples = 1
	batch_size_train = 1
	hidden_size = 2

	shuffle_training_examples = False

	input_size = 1
	output_size = 1

	lr = 0.001
	momentum = 0
	alpha = 0.999
	weight_decay = 0.001

	net = AbsDiagNet(input_size,hidden_size,output_size)
	net.debug_mode = True

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, alpha=alpha, weight_decay=weight_decay)
	
	r1 = javarandom.Random(seed)

	net.recurrent_layer.IH.weight.data[0,0] = r1.nextDouble()
	net.recurrent_layer.IH.weight.data[1,0] = r1.nextDouble()

	net.HO.weight.data[0,0] = r1.nextDouble()
	net.HO.weight.data[0,1] = r1.nextDouble()
	net.HO.bias.data[0] = r1.nextDouble()

	print('----------------------------------------------')
	print('IH:   ' + str(net.recurrent_layer.IH.weight.data))
	print('HH:   ' + str(net.recurrent_layer.HH.data))
	print('HO w: ' + str(net.HO.weight.data))
	print('HO b: ' + str(net.HO.bias.data))

	#net = AbsDiagNetGated(input_size, gate_size, hidden_size, output_size)
	print('using Abs Diag Net')

	r2 = javarandom.Random(seed2)

	sequence_generator = randomSequenceMaker(r2)
	
	trainX, trainY, task_name = batchGenerator(seq_length, batch_size_train, sequence_generator)

	print('task: ' +task_name+', length = ' + str(seq_length))
	print('')

	print('trainX: ' + str(trainX))
	print('trainY: ' + str(trainY))

	print('')
	print('-----------------------------------------')


	###############################################################3
	predictedY = net(trainX)
	loss = loss_fn(predictedY, trainY)

	print('loss = ' + str(loss.item()))
	optimizer.zero_grad()
	loss.backward()

	print('----------------------------------------------')
	print('`IH:   ' + str(net.recurrent_layer.IH.weight.grad.data))
	print('`HH:   ' + str(net.recurrent_layer.HH.grad.data))
	print('`HO w: ' + str(net.HO.weight.grad.data))
	print('`HO b: ' + str(net.HO.bias.grad.data))

	
		
	#Important to apply both kinds of gradient clipping
	norm = torch.nn.utils.clip_grad_norm_(net.parameters(), global_norm_clip)

	print('')
	print('norm = ' + str(norm))

	print('')
	print('post-clip1:')
	print('----------------------------------------------')
	print('`IH:   ' + str(net.recurrent_layer.IH.weight.grad.data))
	print('`HH:   ' + str(net.recurrent_layer.HH.grad.data))
	print('`HO w: ' + str(net.HO.weight.grad.data))
	print('`HO b: ' + str(net.HO.bias.grad.data))

	torch.nn.utils.clip_grad_value_(net.parameters(), local_grad_clip)

	print('')
	print('post-clip2:')
	print('----------------------------------------------')
	print('`IH:   ' + str(net.recurrent_layer.IH.weight.grad.data))
	print('`HH:   ' + str(net.recurrent_layer.HH.grad.data))
	print('`HO w: ' + str(net.HO.weight.grad.data))
	print('`HO b: ' + str(net.HO.bias.grad.data))

	

	optimizer.step()

	net.clamp()

	print('done')