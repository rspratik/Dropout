''''
Simple Standard MLP model with weight/learning decay
Activation Fn: Leaky relu with alpha=0.1
Cost Optimization Fn: Gradient Descent with decay learning.
and weight decay rate = 0.96 and start_learning_rate = 0.5
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

cwd = "./mnist/mlp/standard_decay"
epochs = 50

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

#Model
def model(X, w_h1, w_o, activation, alpha_leak = 0.1):
    with tf.name_scope("Hidden_Layer"):
        if(activation == tf.nn.leaky_relu):
            h = activation(tf.matmul(X, w_h1), alpha=alpha_leak)  # this is a basic mlp
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
    py_x = model(X, w_h1, w_o, activation = tf.nn.leaky_relu)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
        global_step = tf.Variable(0)  # count the number of steps taken.
        start_learning_rate = 0.5
        #exponential_decay(learning_rate, global_step, decay_steps, decay_rate,staircase=False, name=None):
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1000, 0.96, staircase=True)
        #learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
        print('Learning Rate' + str(learning_rate))
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
        #save_path = saver.save(sess, "./mlp/learning_decay/ckpoint/mlp.ckpt")
        #print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    main()
