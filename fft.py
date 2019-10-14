from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend as K
from bits import *
from tensorflow.keras.initializers import Constant

import tensorflow as tf
from tensorflow.keras import layers, regularizers, constraints
import numpy as np
import sys
from scipy.linalg import hadamard
from optparse import OptionParser
import matplotlib.pyplot as plt

EPS = 0.00000001	
parser = OptionParser()
parser.add_option("--n", "--n", dest="n", default="16")
parser.add_option("--num_layers", "--num_layers", dest="num_layers", default="2")
parser.add_option("--C", "--C", dest="C", default="1")
parser.add_option("--eps", "--eps", dest="eps", default="0.01")
parser.add_option("--batch", "--batch", dest="batch", default="32")
parser.add_option("--niter", "--niter", dest="niter", default="100")
parser.add_option("--ntrain", "--ntrain", dest="ntrain", default="256")
parser.add_option("--epochs", "--epochs", dest="epochs", default="100")
parser.add_option("--entropy", "--entropy", dest="entropy", default="False", help="Use entropy regulatizer? Default False (l1 reg)")
parser.add_option("--iniitializer", "--initializer", dest="initializer", default="orthogonal", help="Either orthogonal (initialize with random orthgonal matrix) or opt (hypothesize optimal solution)")
(options, args) = parser.parse_args()
n = int(options.n)
logn = int(np.log(n+0.0001)/np.log(2))
num_layers = int(options.num_layers)
C = float(options.C)
eps = float(options.eps)
batch = int(options.batch)
niter = int(options.niter)
ntrain = int(options.ntrain)
epochs = int(options.epochs)
had = hadamard(n)/np.sqrt(n)
initializer = options.initializer
entropy = eval(options.entropy)

print ("n=%d logn=%d" % (n, logn))

def entropy_reg(C):
	def entropy_reg_do(weight_matrix):
		return  C*K.sum(-(K.square(weight_matrix)+EPS) * K.log(K.square(weight_matrix)+EPS))
	return entropy_reg_do

print ("n=%d" % n)

print ("Computing hypothesized optimum")
# Get hypothesized optimal layers
prod = np.eye(n)
opt_mats = []
opt_layers = []
running_prod = np.eye(n)
mats_per_opt_layer = int(logn / num_layers)
for i in range(logn):
	mat = np.zeros((n,n))
	for j in range(int(n/2)):
		bits0 = int2bitvec(j, logn-1)
		bits1 = int2bitvec(j, logn-1)
		bits0.insert(i, 0)
		bits1.insert(i, 1)
		
		j0 = bitvec2int(bits0)
		j1 = bitvec2int(bits1)
		mat[j0, j0] = 1/np.sqrt(2)
		mat[j0, j1] = 1/np.sqrt(2)
		mat[j1, j0] = 1/np.sqrt(2)
		mat[j1, j1] = -1/np.sqrt(2)
	opt_mats.append(mat)
	running_prod = np.matmul(mat, running_prod)
	if ((i % mats_per_opt_layer) == mats_per_opt_layer - 1):
		opt_layers.append(running_prod)
		running_prod = np.eye(n)
	prod = np.matmul(mat, prod)

print(prod)
print ("##")
print(had)
print ("Hypothesized OPT LAYERS:")
print (opt_layers)
model = tf.keras.Sequential()
for i in range(num_layers):
	if entropy:
		regularizer = entropy_reg(C)
	else:
		regularizer = regularizers.l1(C)

	if initializer == "orthogonal":
		keras_initializer_arg = "orthogonal"
	else:
		print ("Using hypothesized optimal solution for layer %d initializer" % i)
		keras_initializer_arg = Constant(value = opt_layers[i])
	model.add(layers.Dense(
		n, 
		use_bias=False,
		kernel_initializer=keras_initializer_arg,
		kernel_constraint= constraints.unit_norm(axis=1),
		kernel_regularizer=regularizer))


model.compile(
	optimizer = 'adam',
	loss='mse',
	metrics=['mae'])

Xtrain = np.random.normal(size=(ntrain, n))
Ytrain = np.matmul(Xtrain, had)

model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch)

print ("Solution:")
for i in range(num_layers):
	print (model.layers[i].get_weights())