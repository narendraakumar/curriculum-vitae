
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

########################
### Data Processing ####
########################
# Organize the data and feed it to associated dictionaries.
data={}

data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels

# Get only the samples with zero and one label for training.
index_list_train = []
for sample_index in range(data['train/label'].shape[0]):
    label = data['train/label'][sample_index]
    if label == 1 or label == 0:
        index_list_train.append(sample_index)

# Reform the train data structure.
data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]


# Get only the samples with zero and one label for test set.
index_list_test = []
for sample_index in range(data['test/label'].shape[0]):
    label = data['test/label'][sample_index]
    if label == 1 or label == 0:
        index_list_test.append(sample_index)

# Reform the test data structure.
data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]
###############################################
########### Defining place holders ############
###############################################
image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
label_place = tf.placeholder(tf.int32, shape=([None,]), name='gt')
label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
dropout_param = tf.placeholder(tf.float32)

##################################################
########### Model + Loss + Accuracy ##############
##################################################
# A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs = FLAGS.num_classes, scope='fc')
# Define loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
# Accuracy
with tf.name_scope('accuracy'):
    # Evaluate the model
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
    # Accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(accuracy)