import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from optimizer.rmspropclipped import *
import math, random
from models.models import *
from datasets.datasets import *
from pprint import pprint

if __name__ == '__main__':

	print('')
	print('')
	print('-----------------------------------')
	print('RNN demo')

	SANITY_CHECK = False
	seed = 44
	if seed != None:
		random.seed(seed)
		torch.manual_seed(seed)
		print('seed='+str(seed))

	use_abs_diag_rnn = True # False -> use lstm

	global_norm_clip = 30.0 #TODO
	local_grad_clip = 1.0

	seq_length = 500 #TODO
	total_training_examples = 1000 #1000
	show_every = 100 #50
	batch_size_train = 1 #50 #TODO: different than the Java implementation
	batch_size_test = 1000
	gate_size = 20
	hidden_size = 40

	shuffle_training_examples = False

	input_size = 2 #TODO
	output_size = 1
	timesteps = 1000

	learning_rate = 0.001
	alpha = 0.999
	weight_decay = 0.001

	if use_abs_diag_rnn:
		#net = AbsDiagNet(input_size,hidden_size,output_size) #TODO
		net = AbsDiagNetGated(input_size, gate_size, hidden_size, output_size)
		print('using Abs Diag Net')
		pprint(vars(net))
		pprint(vars(net.recurrent_layer))
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

	optimizer = RMSpropclipped(net.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay, clip=local_grad_clip)

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
			norm = torch.nn.utils.clip_grad_norm_(net.parameters(), global_norm_clip)

			optimizer.step()

			if use_abs_diag_rnn:
				net.clamp()

			if s % show_every == 0:
				predictedY = net(testX)
				loss = loss_fn(predictedY, testY)
				print('batch {:d} of {:d}\ttest loss: {:f}'.format(t+1, timesteps, loss.item()))

			s += 1