import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
	temp = input>0
	self._saved_for_backward(temp)
	return temp * input
        # pass

    def backward(self, grad_output):
        '''Your codes here'''
	return grad_output * self._saved_tensor
        # pass


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
	output = 1 / (1 + np.exp(-input))
	self._saved_for_backward(output)
	return output
        # pass

    def backward(self, grad_output):
        '''Your codes here'''
	grad_input = self._saved_tensor * (1 - self._saved_tensor) * grad_output
	# print np.sum(grad_input)
	return grad_input
        # pass

# definition of softmax layer
class Softmax(Layer):
	def __init__(self, name):
		super(Softmax, self).__init__(name)
	
	def forward(self, input):
		temp = np.exp(input)
		N = input.shape[0]
		M = input.shape[1]
		output = np.ndarray(shape = (N, M))
		for n in range(N):
			sum = np.sum(temp[n])
			output[n] = temp[n] / sum
		self._saved_for_backward(output)
		return output
	
	def backward(self, grad_out):
		N = grad_out.shape[0]
		M = grad_out.shape[1]
		grad_in = np.ndarray(shape = (N, M))
		for n in range(N):
			for m in range(M):
				grad_in[n][m] = 0
				for j in range(M):
					grad_in[n][m] += (grad_out[n][j] * ((m == j) - grad_out[n][m]))
		return grad_in

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

	# print "init b =", self.b

    def forward(self, input):
        '''Your codes here'''
	# print "forward input =", input.shape
	N = input.shape[0]
	output = np.ndarray(shape = (N, self.out_num))
	for n in range(N):
		output[n] = np.matmul(input[n], self.W) + self.b
	self._saved_for_backward(np.transpose(input))
	# print "forward output =", output.shape
	return output
        # pass

    def backward(self, grad_output):
        '''Your codes here'''
	# print "backward grad_output =", grad_output.shape
	grad_input = np.matmul(grad_output, np.transpose(self.W))
	self.grad_W = np.matmul(self._saved_tensor, grad_output)
	self.grad_b = np.sum(grad_output, axis = 0)
	'''
	print "backward grad_W =", self.grad_W.shape
	print "backward grad_b =", self.grad_b.shape
	print "backward b =", self.b.shape
	'''
	# print "backward grad_input =", grad_input.shape
	return grad_input
        # pass

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
	# print "training b =", self.b
