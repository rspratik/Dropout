'''
Code referred from medium.com but it has been customized according to personal needs.
--------------------------------------------------------------------------------------
Text classification using Nested dropout (mlp).
Activation Fn: Leaky-Relu with alpha=0.1
Covergence fn: ADAM with learning rate=0.05
dropout p_input=1 ; p_hidden=0.6
'''

import numpy as np
import random
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
import logging

cwd = "./news/mlp/dropout_nested"
# Parameters
learning_rate = 0.05
training_epochs = 20
batch_size = 150
display_step = 1
# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features
n_classes = 5  # ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.religion.misc", "misc.forsale"]
p_keep_input = 1
p_keep_hidden = 0.6

logging.basicConfig()
categories = ["comp.graphics", "sci.space", "rec.sport.baseball", "talk.religion.misc", "misc.forsale"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,download_if_missing=True)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,download_if_missing=True)
print('Total corpus count in training-set:', len(newsgroups_train.data))
print('Total corpus count in test-set:', len(newsgroups_test.data))
vocab = Counter()

#Creating a vocab from train and test data
for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()] += 1

total_words = len(vocab)

#For getting unique index of a word
def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index


word2index = get_word_2_index(vocab)


def get_batch(df, i, batch_size):
    batches = []
    results = []
    texts = df.data[i * batch_size:i * batch_size + batch_size]
    categories = df.target[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
        batches.append(layer)

    for category in categories:
        y = np.zeros((5), dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        elif category == 2:
            y[2] = 1.
        elif category == 3:
            y[3] = 1.
        elif category == 4:
            y[4] = 1.

        results.append(y)

    return np.array(batches), np.array(results)


n_input = total_words  # Words in vocab
input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
output_tensor = tf.placeholder(tf.float32, [None, n_classes], name="output")


def multilayer_perceptron(input_tensor, weights, biases):
    epochDropout = random.getrandbits(1)
    if (epochDropout == 1):
        layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
        layer_1 = tf.nn.leaky_relu(layer_1_addition,alpha=0.1)

        # Hidden layer with leaky_relu activation
        layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
        layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
        layer_2 = tf.nn.leaky_relu(layer_2_addition,alpha=0.1)

        # Output layer
        out_layer_multiplication = tf.matmul(layer_2, weights['out'])
        out_layer_addition = out_layer_multiplication + biases['out']
    else:
        # applying dropout to input layer
        input_tensor = tf.nn.dropout(input_tensor, p_keep_input)

        layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
        layer_1 = tf.nn.leaky_relu(layer_1_addition, alpha=0.1)
        # applying dropout to hidden layer
        layer_1 = tf.nn.dropout(layer_1, p_keep_hidden)

        # Hidden layer with leaky_relu activation
        layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
        layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
        layer_2 = tf.nn.leaky_relu(layer_2_addition, alpha=0.1)
        # applying dropout to hidden layer
        layer_2 = tf.nn.dropout(layer_2, p_keep_hidden)

        # Output layer
        out_layer_multiplication = tf.matmul(layer_2, weights['out'])
        out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# Initializing the variables
init = tf.global_variables_initializer()
# Launching the graph

with tf.name_scope("accuracy"):
    global_step = tf.Variable(0)  # count the number of steps taken.
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))  # Count correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    with tf.summary.FileWriter(cwd, sess.graph) as writer:
        writer = tf.summary.FileWriter(cwd, sess.graph)
        merged = tf.summary.merge_all()

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(newsgroups_train.data) / batch_size)
            # Looping over all batches
            for i in range(total_batch):
                batch_x, batch_y = get_batch(newsgroups_train, i, batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: batch_x, output_tensor: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss={:.9f}".format(avg_cost))
            # Test model
            #correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
            # Calculate accuracy
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            total_test_data = len(newsgroups_test.target)
            batch_x_test, batch_y_test = get_batch(newsgroups_test, 0, total_test_data)
            summary, acc = sess.run([merged, accuracy], feed_dict={input_tensor: batch_x_test, output_tensor: batch_y_test})
            #print("Accuracy:",acc)
            print("Accuracy :", acc)
            #print("Accuracy3:", summary)
            writer.add_summary(summary, epoch)  # Write summary
        print("Optimization Finished!")
