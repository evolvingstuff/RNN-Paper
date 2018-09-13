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

seq_len, batch_size, hidden_size = 10, 1, 2
size = (seq_len, batch_size, hidden_size)
X = Variable(torch.rand(size), requires_grad=True)
print(X.size())

net = AbsDiagNet(hidden_size, hidden_size, 1, 1.0)
#net = LSTMNet(hidden_size, hidden_size, 1)

from torch.autograd import gradcheck
inputs = [X,]
test = gradcheck(net, inputs)
print(test)