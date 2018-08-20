import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from optimizer.rmspropclipped import *
import math, random
from models.models import *
from datasets.datasets import *
from pprint import pprint
import argparse
import time

if __name__ == '__main__':

	print('')
	print('')
	print('-----------------------------------')
	print('RNN demo')

	parser = argparse.ArgumentParser(description='PyTorch Example')

	SANITY_CHECK = False
	seed = 60 #53
	if seed != None:
		random.seed(seed)
		torch.manual_seed(seed)
		print('seed='+str(seed))

	use_abs_diag_rnn = True # False -> use lstm

	global_norm_clip = 30.0 #30.0 #TODO: figure out how to set this automatically
	local_grad_clip = 1.0

	seq_length = 1000 #1000
	total_training_examples = 250 # 1000
	show_every = 10
	batch_size_train = 64 #50
	batch_size_test = 1000
	gate_size = 20 #20
	hidden_size = 40 # 40

	shuffle_training_examples = True

	init_weight_recur = 1.0

	input_size = 2 #TODO
	output_size = 1
	timesteps = 1000

	learning_rate = 0.001 #0.001
	alpha = 0.999
	weight_decay = 0.001 #0.001

	if use_abs_diag_rnn:
		#net = Diagnet(input_size,hidden_size,output_size, init_weight_recur) #TODO
		net = DiagnetGated(input_size, gate_size, hidden_size, output_size, init_weight_recur)
		print('using Diagnet')
		#pprint(vars(net))
		#pprint(vars(net.recurrent_layer))
	else:
		net = LSTMNet(input_size, hidden_size, output_size, init_weight_recur)
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

	training_examples = []
	for example in range(total_training_examples):
		trainX, trainY, _ = batchGenerator(seq_length, batch_size_train, sequence_generator)
		training_examples.append((trainX, trainY))

	low_loss = 10000000

	for t in range(timesteps):

		if shuffle_training_examples:
			random.shuffle(training_examples)

		s = 0

		current_milli_time = lambda: int(round(time.time() * 1000))

		start = current_milli_time()

		for example in training_examples:

			trainX, trainY = example[0], example[1]
			predictedY = net(trainX)
			loss = loss_fn(predictedY, trainY)

			optimizer.zero_grad()

			loss.backward()
			
			#Constraint #1: Norm of entire gradient is clipped
			norm = torch.nn.utils.clip_grad_norm_(net.parameters(), global_norm_clip)

			#Constraint #2: Element-wise gradient clipping is applied inside this altered optimizer step
			optimizer.step()

			#Constraint #3: Recurrent weights are clipped to range of [-1, 1]
			if use_abs_diag_rnn:
				net.clamp()

			if s % show_every == show_every-1:
				predictedY = net(testX)
				loss = loss_fn(predictedY, testY)
				end = current_milli_time()
				if loss < low_loss:
					print('step {:d} of {:d}\t{:d}ms\ttest loss: {:f} +'.format(t+1, timesteps, end-start, loss.item()))
					low_loss = loss
				else:
					print('step {:d} of {:d}\t{:d}ms\ttest loss: {:f}'.format(t+1, timesteps, end-start, loss.item()))
				start = current_milli_time()

			s += 1