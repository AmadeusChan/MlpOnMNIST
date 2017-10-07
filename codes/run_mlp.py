from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Softmax
from loss import EuclideanLoss, CrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

import json
import sys
import os

import numpy as np
from scipy import misc

def getNetwork():
	'''
	to obtain network structure from specified file
	'''
	file_name = "structure.json"
	if len(sys.argv)>1:
		file_name = sys.argv[1]
	f = file(file_name, "r")
	s = f.read()
	f.close()

	networks = json.loads(s)
	for network in networks:
		config = network['config']
		dis_model = network['model']
		model = Network()
		for layer in dis_model:
			if layer['type'] == 'Linear':
				model.add(Linear(layer['name'], layer['in_num'], layer['out_num'], layer['std']))
			if layer['type'] == 'Relu':
				model.add(Relu(layer['name']))
			if layer['type'] == 'Sigmoid':
				model.add(Sigmoid(layer['name']))
			if layer['type'] == 'Softmax':
				model.add(Softmax(layer['name']))
		loss = EuclideanLoss('loss')
		if 'loss' in config:
			if config['loss'] == 'CrossEntropyLoss':
				loss = CrossEntropyLoss('loss')
		yield network['name'], model, config, loss

train_data, test_data, train_label, test_label = load_mnist_2d('data')
N = train_data.shape[0]

'''
train_data = np.append(train_data, train_data, axis=0) 
for n in range(N, 2*N): 
	image = np.reshape(train_data[n], (28, 28))

	# train_data[n-N] = np.reshape( misc.imrotate(image, 0), (1, 784) ) / 255.0
	image = misc.imrotate(image, 10*np.random.randn()) / 255.0
	train_data[n] = np.reshape(image, (1, 784))


train_label = np.append(train_label, train_label, axis=0)
'''
print train_data.shape

# Your model defintion here
# You should explore different model architecture

# ATTENTION: Now the networks are defined in structure.json (or some other json file) in json format

'''
model = Network()
model.add(Linear('fc1', 784, 196, 0.01))
model.add(Relu('af1'))
model.add(Linear('fc2', 196, 28, 0.01))
model.add(Relu('af2'))
model.add(Linear('fc3', 28, 10, 0.01))
'''

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

'''
config = {
    'learning_rate': 0.001,
    'weight_decay': 0,
    'momentum': 0,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 1
}
'''
acc_file = "results/accuracy.txt"
loss_file = "results/loss.txt"
if len(sys.argv)>2:
	acc_file = sys.argv[2]
if len(sys.argv)>3:
	loss_file = sys.argv[3]

os.system("rm " + acc_file)
os.system("touch " + acc_file)
os.system("rm " + loss_file)
os.system("touch " + loss_file)

for name, model, config, loss in getNetwork():
	
	print type(loss)
	
	outf = file(acc_file, "a")
	outf.write('Network Name: ' + name + '\n')
	outf.close()

	outf = file(loss_file, "a")
	outf.write('Network Name: ' + name + '\n')
	outf.close()

	for epoch in range(config['max_epoch']):
    		LOG_INFO('Network %s: Training @ %d epoch...' % (name, epoch))
   		train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], loss_file)

		if epoch % config['test_epoch'] == 0:
        		LOG_INFO('Network %s: Testing @ %d epoch...' % (name, epoch))
        		accu = test_net(model, loss, test_data, test_label, config['batch_size'])

			outf = file(acc_file, "a")
			outf.write(str(epoch) + ' ' + str(accu) + '\n')
			outf.close()

	outf = file(acc_file, "a")
	outf.write('\n')
	outf.close()

	outf = file(loss_file, "a")
	outf.write('\n')
	outf.close()
