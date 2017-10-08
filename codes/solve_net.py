from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq, loss_file):

    iter_counter = 0
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

	# print "Debug: ", "input=", input.shape, " target=", target.shape

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target) 
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

	'''
	outf = file(loss_file, "a")
	outf.write(str(loss_value) + ' ' + str(acc_value) + '\n')
	outf.close()
	'''

        if iter_counter % disp_freq == 0:
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (iter_counter, np.mean(loss_list), np.mean(acc_list))

	    outf = file(loss_file, "a")
	    outf.write(str(np.mean(loss_list)) + " " + str(np.mean(acc_list)) + '\n')
	    outf.close()

            loss_list = []
            acc_list = []
            LOG_INFO(msg)


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)

    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (np.mean(loss_list), np.mean(acc_list))
    LOG_INFO(msg)
    return np.mean(acc_list)
