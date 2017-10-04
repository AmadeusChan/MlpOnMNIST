from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Softmax
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

import json
import sys
import os

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
		yield network['name'], model, config

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture

# Now the networks are defined in structure.json in json format

'''
model = Network()
model.add(Linear('fc1', 784, 196, 0.01))
model.add(Relu('af1'))
model.add(Linear('fc2', 196, 28, 0.01))
model.add(Relu('af2'))
model.add(Linear('fc3', 28, 10, 0.01))
'''

loss = EuclideanLoss(name='loss')

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
acc_file = "accuracy.txt"
if len(sys.argv)>2:
	acc_file = sys.argv[2]

os.system("rm " + acc_file)
os.system("touch " + acc_file)

for name, model, config in getNetwork():
	
	outf = file(acc_file, "a")
	outf.write('Network Name: ' + name + '\n')
	outf.close()

	for epoch in range(config['max_epoch']):
    		LOG_INFO('Network %s: Training @ %d epoch...' % (name, epoch))
   		train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

		if epoch % config['test_epoch'] == 0:
        		LOG_INFO('Network %s: Testing @ %d epoch...' % (name, epoch))
        		accu = test_net(model, loss, test_data, test_label, config['batch_size'])

			outf = file(acc_file, "a")
			outf.write(str(epoch) + ' ' + str(accu) + '\n')
			outf.close()

	outf = file(acc_file, "a")
	outf.write('\n')
	outf.close()

