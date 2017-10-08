from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
	temp = input - target
	# print "input =", input
	return np.sum(temp * temp) / 2 / target.shape[0]
        # pass

    def backward(self, input, target):
        '''Your codes here'''
	return (input - target) / target.shape[0]
        # pass

class CrossEntropyLoss(object):
	esp = 10e-10

	def __init__(self, name):
		self.name = name
	
	def forward(self, input, target):
		return np.sum(-target * np.log(input))
	
	def backward(self, input, target):
		return -target / (input)
