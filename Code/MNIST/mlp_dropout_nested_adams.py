''''
Nested Dropout MLP model: Dropout over the hidden layers would be applied stochastically.
Activation Fn: Leaky relu with alpha=0.1
Cost Optimization Fn: ADAMS learning rate =0.05
dropout p_input=1 ; p_hidden=0.6
'''

import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

cwd = "./mnist/mlp/dropout_nested_adams"
epochs = 50
p_keep_input = 1
p_keep_hidden = 0.6


def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

#Model
def model(X, w_h1, w_o, p_keep_input, p_keep_hidden, activation, alpha_leak = 0.1):
    with tf.name_scope("Hidden_Layer"):
        print("Randomizing the epoch dropout")
        #Randomizing the epoch dropout
        epochDropout = random.getrandbits(1)
        if(epochDropout == 1):
            #Applying dropout on input layer
            X = tf.nn.dropout(X, keep_prob = p_keep_input)
            if(activation == tf.nn.leaky_relu):
                h = activation(tf.matmul(X, w_h1), alpha=alpha_leak)
            else:
                h = activation(tf.matmul(X, w_h1))
            # Applying dropout on hidden layer
            h = tf.nn.dropout(h,keep_prob = p_keep_hidden)
        else:
            if (activation == tf.nn.leaky_relu):
                h = activation(tf.matmul(X, w_h1), alpha=alpha_leak)
            else:
                h = activation(tf.matmul(X, w_h1))
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    size_h1 = tf.constant(625, dtype=tf.int32)

    X = tf.placeholder("float", [None, 784], name="X")
    Y = tf.placeholder("float", [None, 10], name="Y")

    w_h1 = init_weights([784, size_h1], "w_h") # create symbolic variables
    w_o = init_weights([size_h1, 10], "w_o")

    #Saving weights
    tf.summary.histogram("w_h1", w_h1)
    tf.summary.histogram("w_o1", w_o)


    #tf.nn.relu
    #tf.nn.leaky_relu
    #tf.nn.sigmoid
    py_x = model(X, w_h1, w_o, p_keep_input, p_keep_hidden, activation = tf.nn.leaky_relu)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
        train_op = tf.train.AdamOptimizer(0.05).minimize(cost) # construct an optimizer
        tf.summary.scalar("cost", cost)

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        # Add scalar summary for accuracy tensor
        tf.summary.scalar("accuracy", acc_op)

    # Saving
    #saver = tf.train.Saver()

    with tf.Session() as sess:
        # tensorboard --logdir=./logs/mlp-2a
        # Use different dir for writing diff models so that comparison go easy.
        with tf.summary.FileWriter(cwd, sess.graph) as writer:
            writer = tf.summary.FileWriter(cwd, sess.graph)
            merged = tf.summary.merge_all()

            tf.global_variables_initializer().run()

            for i in range(epochs):
                for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY})
                writer.add_summary(summary, i)  # Write summary
                print(i, acc)                   # Report the accuracy

        # Save the variables to disk.
        #save_path = saver.save(sess, "./mlp/alternating_dropout/ckpoint/mlp.ckpt")
        #print("Model saved in file: %s" % save_path)

if __name__ == "__main__":
    main()
