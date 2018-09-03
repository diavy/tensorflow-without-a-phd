########## Deep 5-layers NN built for digit recognition ###################


############ Import modules #################
import math

import mnistdata
import tensorflow as tf

tf.set_random_seed(0)


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)


########### Define input, label, training parameters (weight and bias) ################

# input X: 28 * 28 gray-scale pixels
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_digits_2D')
# label Y_: the expected label
Y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='truth_label')

# variable learning rate
#lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)

#### three convolutional layers and their channel counts respectively. and a fully connected layer
# THe last layer has 10 softmax neurons #######
K = 6 # first convolutional layer output depth
L = 12 # second convolutional layer output depth
M = 24 # thir convolutional layer output depth
N = 200 # fully connected layer
######## five filter weights matrix ###########
W1 = tf.Variable(initial_value=tf.truncated_normal([6, 6, 1, K], stddev=0.1)) # 5*5 patch, 1 input channel, K output channels
W2 = tf.Variable(initial_value=tf.truncated_normal([5, 5, K, L], stddev=0.1))
W3 = tf.Variable(initial_value=tf.truncated_normal([4, 4, L, M], stddev=0.1))
W4 = tf.Variable(initial_value=tf.truncated_normal([7 * 7 * M, N], stddev=0.1)) # fully connected layer
W5 = tf.Variable(initial_value=tf.truncated_normal([N, 10], stddev=0.1))
######## five bias vectors #############

# B1 = tf.Variable(initial_value=tf.zeros([L]))
# B2 = tf.Variable(initial_value=tf.zeros([M]))
# B3 = tf.Variable(initial_value=tf.zeros([N]))
# B4 = tf.Variable(initial_value=tf.zeros([O]))
# B5 = tf.Variable(initial_value=tf.zeros([10]))

# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
B1 = tf.Variable(initial_value=tf.ones([K])/10)
B2 = tf.Variable(initial_value=tf.ones([L])/10)
B3 = tf.Variable(initial_value=tf.ones([M])/10)
B4 = tf.Variable(initial_value=tf.ones([N])/10)
B5 = tf.Variable(initial_value=tf.ones([10])/10)

########### Define model, expected output, loss function and training method ###########

# the model
stride = 1 ## output is 28*28*K
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2 ## output is 14*14*L
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2 ## output is 7*7*M
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the thrid convolutional layer to the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7*7*M])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep) # add dropout to fully connected layer
Ylogits = tf.matmul(YY4, W5) + B5
Y = tf.nn.softmax(Ylogits)


# loss function: use built-in cross entropy with logits function to avoid log(0) if it happens
#cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100  # make it comparable with test dataset with 10000 samples
#cross_entropy = tf.reduce_sum(cross_entropy) / 10  # this should be equal to above

# optimizer set up: GradientDecent(mini-batch), learning rate is 0.003
# lr = 0.003
# using decay learning rate
# step for variable learning rate

#step = tf.placeholder(tf.int32)
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss = cross_entropy)
# accuracy of the prediction
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#########################################################################################

###########################  set up training process ####################
## initialization
init = tf.global_variables_initializer()  #
sess = tf.Session()
sess.run(init)
## batch training actions definition ##
def do_training(i):
    batch_X, batch_Y = mnist.train.next_batch(100)  # get batch-size data

    ##### print training accuracy and loss ############
    train_a, train_c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_:batch_Y, pkeep:1.0})
    print("training " + str(i) + ": accuracy: " + str(train_a) + " loss: " + str(train_c))

    ##### print testing accuracy and loss ##############
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_:mnist.test.labels, pkeep:1.0})
    print("testing " + str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) +
          " ********* test accuracy:" + str(test_a) + " test loss: " + str(test_c))

    ##### backpropagation training ########
    sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y, step:i, pkeep:0.75}) ### drop out at training stage

with sess:
    iterations = 2000
    for i in range(iterations):
        do_training(i)

    print("#############final performance on testing data###############")
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    print("accuracy:" + str(test_a) + " loss: " + str(test_c))


##### print testing accuracy and loss ##############







