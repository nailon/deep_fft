from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend as K

import tensorflow as tf
from tensorflow.keras import layers, regularizers, constraints
import numpy as np
import sys
from scipy.linalg import hadamard
from optparse import OptionParser
import matplotlib.pyplot as plt

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
entropy = eval(options.entropy)

print ("n=%d logn=%d" % (n, logn))


def entropy_reg(C):
	def entropy_reg_do(weight_matrix):
		return  C*K.sum(-K.square(weight_matrix) * K.log(K.square(weight_matrix)))
	return entropy_reg_do

print ("n=%d" % n)

model = tf.keras.Sequential()
for i in range(num_layers):
	if entropy:
		regularizer = entropy_reg(C)
	else:
		regularizer = regularizers.l1(C)
	model.add(layers.Dense(
        n, 
        use_bias=False,
        kernel_initializer="orthogonal",
	    kernel_constraint= constraints.unit_norm(axis=1),
        kernel_regularizer=regularizer))


model.compile(
	optimizer = 'adam',
	loss='mse',
	metrics=['mae'])

Xtrain = np.random.normal(size=(ntrain, n))
Ytrain = np.matmul(Xtrain, had)

model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch)

prod = np.eye(n)
for i in range(num_layers):
	prod = np.matmul(prod, model.layers[i].get_weights()[0])
	weights = np.square(model.layers[i].get_weights()[0])
	for j in range(n):
		weights[j,:] = weights[j,:]  / np.max(weights[j,:])
	plt.imshow(weights, cmap="gray")
	plt.savefig("heatmap-%d.png" % i)
	print ("Layer %d square-weights: (normalied by max in each row)" % i)
	print (weights)

sqrprod = np.square(prod)
for j in range(n):
	sqrprod[j,:] = sqrprod[j,:]  / np.max(sqrprod[j,:])
plt.imshow(sqrprod, cmap = "gray")
plt.savefig("prod.png")
print ("prod=")
print (sqrprod)
# Create heatmap of weight matrix - first layer
print ("Shape of prod matrix:")
print (prod.shape)
print ("Row norms of weight matrices:")
for i in range(num_layers):
	print("layer %d norm %s" % (i, np.linalg.norm(model.layers[i].get_weights()[0], axis=1)))

#print ("prod:")
#print (prod)
#print ("Dimension of product matrix:")
#print (prod.shape)
#print ("prod=", prod)
#print ("had=", had)
pred = np.matmul(Xtrain, prod)
#print ("First 5 labels:")
#print (Ytrain[:5,:])
#print ("First 5 predictions:")
#print (pred[:5,:])
residue = Ytrain - np.matmul(Xtrain, prod)
#print ("Deminsions of residue:")
#print (residue.shape)
err_vec = np.mean(np.square(residue), axis=1)
#print ("First 5 coordinates of err_vec:")
#print (err_vec[:5])
print ("err=%f" % np.mean(err_vec))

#print (model.layers[0].get_weights())
tot_reg = 0.0
for i in range(num_layers):
	tot_reg = tot_reg + np.sum(np.abs(model.layers[i].get_weights()))

print ("tot_reg = %f" % tot_reg)

