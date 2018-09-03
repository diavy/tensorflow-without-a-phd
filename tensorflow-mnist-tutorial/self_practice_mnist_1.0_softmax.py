########## Single softmax layer NN build for digit recognition ###################


############ Import modules #################
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
# weights matrix W[28*28, 10]
W = tf.Variable(initial_value=tf.zeros([28*28, 10]))
# bias b[10]
b = tf.Variable(initial_value=tf.zeros([10]))

########### Define model, expected output, loss function and training method ###########
## flatten the 28*28 image into 1 single line of pixels
XX = tf.reshape(X, shape=[-1, 28*28])
# the softmax model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
# loss function
cross_entropy = - tf.reduce_sum(Y_ * tf.log(Y))
# optimizer set up: GradientDecent(mini-batch), learning rate is 0.005
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss = cross_entropy)
# accuracy of the prediction
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#########################################################################################

###########################  set up training process ####################
## initilization
init = tf.global_variables_initializer()  #
sess = tf.Session()
sess.run(init)
## batch training actions definition ##
def do_training(i):
    batch_X, batch_Y = mnist.train.next_batch(100)  # get batch-size data

    ##### print training accuracy and loss ############
    train_a, train_c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_:batch_Y})
    print("training " + str(i) + ": accuracy: " + str(train_a) + " loss: " + str(train_c))

    ##### print testing accuracy and loss ##############
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_:mnist.test.labels})
    print("testing " + str(i) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) +
          " ********* test accuracy:" + str(test_a) + " test loss: " + str(test_c))

    ##### backpropagation training ########
    sess.run(train_step, feed_dict={X:batch_X, Y_:batch_Y})

with sess:
    iterations = 1000
    for i in range(iterations):
        do_training(i)

    print("#############final performance on testing data###############")
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    print("accuracy:" + str(test_a) + " loss: " + str(test_c))


##### print testing accuracy and loss ##############







