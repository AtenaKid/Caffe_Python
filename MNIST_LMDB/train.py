# This python implementation is follow the jupyter example in 
# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

import caffe
import numpy as np
import os
import sys
#import cv2
from pylab import *
import matplotlib.pyplot as plt

# init
caffe.set_device(0)
caffe.set_mode_gpu()

### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')

print ' Finish loading solver '

# Checking some condition 
# each output is (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (we'll omit the biases)
print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

# Before taking off, let's check that everything is loaded as we expect. We'll run a forward pass on the train and test nets and check that they contain our data.
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one) 

# we use a little trick to tile the first eight images
# For training 
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off'); plt.show()
print 'train labels:', solver.net.blobs['label'].data[:8]

# For testing
plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray'); axis('off'); plt.show()
print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]

# Steping the solver. 
# Let's take one step of (minibatch) SGD and see what happens.
solver.step(1)
# Do we have gradients propagating through our filters? Let's see the updates to the first layer, 
# shown here as a 4×54×5 grid of 5×55×5 filters.
plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray'); axis('off'); plt.show()

# === Writing a custom training loop ============================================================================
#   Something is happening. Let's run the net for a while, keeping track of a few things as it goes. 
#   Note that this process will be the same as if training through the caffe binary. In particular:
#       - logging will continue to happen as normal
#       - snapshots will be taken at the interval specified in the solver prototxt (here, every 5000 iterations)
#       - testing will happen at the interval specified (here, every 500 iterations)
#   Since we have control of the loop in Python, we're free to compute additional things as we go, as we show below. We can do many other things as well, for example:
#       - write a custom stopping criterion
#       - change the solving process by updating the net in the loop

#time
niter = 200
test_interval = 25
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:       
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
        print 'Iteration', it, 'testing...' , 'Accuracy: ', (correct/1e4)

# Let's plot the train loss and test accuracy 
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.show()


# The loss seems to have dropped quickly and coverged (except for stochasticity), while the accuracy rose correspondingly. Hooray!
# Since we saved the results on the first test batch, we can watch how our prediction scores evolved. We'll plot time on the xx axis 
# and each possible label on the yy, with lightness indicating confidence.

for i in range(8):
    figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray'); plt.show()
    figure(figsize=(10, 2))
    plt.imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
    plt.show()

# We started with little idea about any of these digits, and ended up with correct classifications for each. If you've been following along, 
# you'll see the last digit is the most difficult, a slanted "9" that's (understandably) most confused with "4".
# Note that these are the "raw" output scores rather than the softmax-computed probability vectors. The latter,
# shown below, make it easier to see the confidence of our net (but harder to see the scores for less likely digits).

for i in range(8):
    figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray'); plt.show()
    figure(figsize=(10, 2))
    plt.imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    xlabel('iteration')
    ylabel('label')
    plt.show()