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

#### five layers and their number of neurons respectively. THe last layer has 10 softmax neurons #######
L, M, N, O = 200, 100, 60, 30
######## five weights matrix ###########
W1 = tf.Variable(initial_value=tf.truncated_normal([28*28, L], stddev=0.1))
W2 = tf.Variable(initial_value=tf.truncated_normal([L, M], stddev=0.1))
W3 = tf.Variable(initial_value=tf.truncated_normal([M, N], stddev=0.1))
W4 = tf.Variable(initial_value=tf.truncated_normal([N, O], stddev=0.1))
W5 = tf.Variable(initial_value=tf.truncated_normal([O, 10], stddev=0.1))
######## five bias vectors #############

# B1 = tf.Variable(initial_value=tf.zeros([L]))
# B2 = tf.Variable(initial_value=tf.zeros([M]))
# B3 = tf.Variable(initial_value=tf.zeros([N]))
# B4 = tf.Variable(initial_value=tf.zeros([O]))
# B5 = tf.Variable(initial_value=tf.zeros([10]))

# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
B1 = tf.Variable(initial_value=tf.ones([L])/10)
B2 = tf.Variable(initial_value=tf.ones([M])/10)
B3 = tf.Variable(initial_value=tf.ones([N])/10)
B4 = tf.Variable(initial_value=tf.ones([O])/10)
B5 = tf.Variable(initial_value=tf.zeros([10]))

########### Define model, expected output, loss function and training method ###########
## flatten the 28*28 image into 1 single line of pixels
XX = tf.reshape(X, shape=[-1, 28*28])

# the model
# Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)

# Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
# Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
# Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
# Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
#
# Ylogits = tf.matmul(Y4, W5) + B5

### add drop out ###########

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5

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
    train_a, train_c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_:batch_Y,
                                                                      step:i, pkeep:1.0})
    print("training " + str(i) + ": accuracy: " + str(train_a) + " loss: " + str(train_c))

    ##### print testing accuracy and loss ##############
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_:mnist.test.labels,
                                                                    step:i, pkeep:1.0})
    print("testing " + str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) +
          " ********* test accuracy:" + str(test_a) + " test loss: " + str(test_c))

    ##### backpropagation training ########
    sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y, step:i, pkeep:0.75}) ### drop out at training stage

with sess:
    iterations = 5000
    for i in range(iterations):
        do_training(i)

    print("#############final performance on testing data###############")
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels,
                                                                    step:iterations, pkeep:1.0})
    print("accuracy:" + str(test_a) + " loss: " + str(test_c))


##### print testing accuracy and loss ##############







