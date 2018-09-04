########################## self-practice of RNN with GRU cell construction #################


################## import necessary modules ##########################
import os

import my_txtutils as txt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1

tf.set_random_seed(0)

# model parameters
#
# Usage:
#   Training only:
#         Leave all the parameters as they are
#         Disable validation to run a bit faster (set validation=False below)
#         You can follow progress in Tensorboard: tensorboard --log-dir=log
#   Training and experimentation (default):
#         Keep validation enabled
#         You can now play with the parameters anf follow the effects in Tensorboard
#         A good choice of parameters ensures that the testing and validation curves stay close
#         To see the curves drift apart ("overfitting") try to use an insufficient amount of
#         training data (shakedir = "shakespeare/t*.txt" for example)
#


########## define some global parameters #########
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE  ## all characters vocabulary size, also the unit input size x
INTERNALSIZE = 512   ## hidden unit size, will be concatenated to unit input size x
NLAYERS = 3 ## number of hidden state layers, the hidden unit size is the same across all layers
learning_rate = 0.001 # fixed learning rate
dropout_pkeep = 0.8 # some dropout to avoid outfitting

########## load data and split them into training and validation sets#################
shakedir = "shakespeare/*.txt"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN) # number epoch to finish the entire training set
txt.print_data_stats(len(codetext), len(valitext), epoch_size)


############################### The model definition ################################
# parameters
lr = tf.placeholder(tf.float32, name='learning_rate')  # learning rate
pkeep = tf.placeholder(tf.float32, name='dropout_probability') # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')


# inputs
X = tf.placeholder(tf.uint8, shape=[None, None], name='X')   # [BATCHSIZE, SEQLEN]
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)     # [BATCHSIZE, SEQLEN, ALPHASIZE]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, shape=[None, None], name="Y_")   # [BATCHSIZE, SEQLEN]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)    # [BATCHSIZE, SQELEN, ALPHASIZE]
# state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='hidden_state') # [BATCHSIZE, INTERNALSIZE*NLAYES]

### using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
### dynamic_rnn infers SEQLEN from the size of inputs Xo
### apply dropout in RNNs
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False) # set state_is_tuple as False to concatenate all hidden states
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep) # dropout for the softmax layer

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [BATCHSIZE, SQELEN, INTERNALSIZE]
# H: [BATCHSIZE, INTERNALSIZE*NLAYERS]  the last state in the sequence
H = tf.identity(H, name='H')



##### softmax layer implementation ########
## Flatten the first two dimension of the output [BATCHSIZE, SEQLEN, INTERNALSIZE] => [BATCHSIZE*SEQLEN, INTERNALSIZE]
## then apply softmax readout layer. This way, the weights and biases are shared across all unrolled time steps.
## From the readout point of view, a value coming from a sequence time step or a minibatch item is the same

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])     # [BATCHSIZE*SEQLEN, INTERNALSIZE]
Ylogits = layers.linear(Yflat, ALPHASIZE)      # [BATCHSIZE*SEQLEN, ALPHASIZE]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])      # [BATCHSIZE*SEQLEN, ALPHASIZE]

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Yflat_) # [BATCHSIZE * SEQLEN]
loss = tf.reshape(loss, [batchsize, -1])    # [BATCHSIZE, SEQLEN]
Yo = tf.nn.softmax(Ylogits, name='Yo')      # output from model [BATCHSIZE*SEQLEN, ALPHASIZE]
Y = tf.argmax(Yo, 1)   # [BATCHSIZE*SEQLEN]
Y = tf.reshape(Y, [batchsize, -1], name='Y')  # [BATCHSIZE, SEQLEN]

### training function with BP ###
## loss = tf.reduce_mean(loss) * BATCHSIZE ### do we need this step ??
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

### stats for display ######
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")


###############################run the model training session #################################
init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)

def do_training():
    istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])  # initial zero input state, this is a must
    step = 0
    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):

        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin : istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
        _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)

        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            txt.print_text_generation_header()
            #### shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            ry = np.array([[txt.convert_from_alphabet(ord("K"))]])  ### why this? random picked?
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            for k in range(1000):  # output 1000 characters
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
                rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(chr(txt.convert_to_alphabet(rc)), end="")
                ry = np.array([[rc]]) ## use the current output to feed the next stage input

            txt.print_text_generation_footer()


        # save a checkpoint (every 500 batches)
        # if step // 10 % _50_BATCHES == 0:
        #     saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        #     print("Saved file: " + saved_file)

        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)

        # loop state around
        istate = ostate # wire the current output state to next input state
        step += BATCHSIZE * SEQLEN # move on to next batch


with tf.Session() as sess:
    sess.run(init)
    do_training()






# all runs: SEQLEN = 30, BATCHSIZE = 100, ALPHASIZE = 98, INTERNALSIZE = 512, NLAYERS = 3
# run 1477669632 decaying learning rate 0.001-0.0001-1e7 dropout 0.5: not good
# run 1477670023 lr=0.001 no dropout: very good

# Tensorflow runs:
# 1485434262
#   trained on shakespeare/t*.txt only. Validation on 1K sequences
#   validation loss goes up from step 5M (overfitting because of small dataset)
# 1485436038
#   trained on shakespeare/t*.txt only. Validation on 5K sequences
#   On 5K sequences validation accuracy is slightly higher and loss slightly lower
#   => sequence breaks do introduce inaccuracies but the effect is small
# 1485437956
#   Trained on shakespeare/*.txt. Validation on 1K sequences
#   On this much larger dataset, validation loss still decreasing after 6 epochs (step 35M)
# 1495447371
#   Trained on shakespeare/*.txt no dropout, 30 epochs
#   Validation loss starts going up after 10 epochs (overfitting)
# 1495440473
#   Trained on shakespeare/*.txt "naive dropout" pkeep=0.8, 30 epochs
#   Dropout brings the validation loss under control, preventing it from
#   going up but the effect is small.
