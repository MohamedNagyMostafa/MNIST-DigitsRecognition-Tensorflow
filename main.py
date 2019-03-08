import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


#parameters
learning_rate = 0.00001
epochs = 5
batch_size = 280

test_valid_size = 256

#network parameters
n_classes = 10
dropout = 0.75

#convolution parameter
filter_size_height = 5
filter_size_width = 5
color_channels = 1
output_depth_cv1 = 32
output_depth_cv2 = 64

#fully-connecteed
hidden_layer_nodes = 1024
output_layer_nodes = n_classes


 
#Convolution layers

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize= [1, k, k, 1],
        strides = [1, k, k, 1],
        padding = 'SAME')

def conv_net(x, weights, biasses, dropout):
    #layer 1 28 x 28 x 1 -> 14 x 14 x 32
    conv1 = conv2d(x, weights['wc1'], biasses['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    #layer2 14 x 14 x 32 -> 7 x 7 x 64
    conv2 = conv2d(conv1, weights['wc2'], biasses['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    #fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biasses['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['wd2']), biasses['bd2'])

    return out

mnist = input_data.read_data_sets(".", one_hot= True, reshape=False)

# store weight and bias
weights = {
    'wc1': tf.Variable(tf.random_normal([filter_size_height, filter_size_width, color_channels, output_depth_cv1])),
    'wc2': tf.Variable(tf.random_normal([filter_size_width, filter_size_height, output_depth_cv1, output_depth_cv2])),
    'wd1': tf.Variable(tf.random_normal([7*7*output_depth_cv2, hidden_layer_nodes])),
    'wd2': tf.Variable(tf.random_normal([hidden_layer_nodes, output_layer_nodes]))
}

biasses= {
    'bc1': tf.Variable(tf.random_normal([output_depth_cv1])),
    'bc2': tf.Variable(tf.random_normal([output_depth_cv2])),
    'bd1': tf.Variable(tf.random_normal([hidden_layer_nodes])),
    'bd2': tf.Variable(tf.random_normal([output_layer_nodes]))
}
#tensorflow graph
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

with tf.device('/gpu:0'):
    
    #Model
    logits = conv_net(x, weights, biasses, keep_prob)

    #Optimizer & Cost

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    #Accuracy
    correct_pro = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pro, tf.float32))

# Initialization variables
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)) as session:
    session.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            session.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})
            #Calculate accuracy

            loss = session.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = session.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})
            
            print('Epoch {:>2}, Batch {:>3} -Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))
                
                
    # Calculate Test Accuracy
    test_acc = session.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))

































