import sys, time, os, pickle, numpy as np, tensorflow as tf
from collections import namedtuple
from tqdm import tqdm 



batch_size = 10000

train_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]


# function provided in the official website
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_train_batch(i):
	data = unpickle(os.path.join("..","cifar-10-batches-py", train_files[i]))
	x = data[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1) # this transpose helps to fit the data with tf.conv2d function check out []
	y = data[b'labels']
	return x, y


def onehot(x):
	return tf.one_hot(x, 10)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def build_architecture():
	tf.reset_default_graph()

	with tf.name_scope('inputs'):
		inputs = tf.placeholder(tf.float32, shape=[batch_size,32,32,3], name='inputs')

	with tf.name_scope('labels'):
		labels = tf.placeholder(tf.float32, shape=[batch_size,10], name='labels')

	keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	with tf.name_scope('Convolution1'):
		w1 = tf.get_variable(name='w1', shape=[3,3,3,10], initializer=tf.initializers.truncated_normal(-0.1, 0.1)) # this will be the kernel
		b1 = tf.get_variable(name='b1', shape=[10], initializer=tf.zeros_initializer())

		c1 = tf.nn.leaky_relu(tf.add(conv2d(inputs, w1), b1))

	with tf.name_scope('Pooling1'):
		c1 = max_pool(c1)
	# print('====c1 shape:', c1.get_shape())	
	with tf.name_scope('Flatten'):
		c1 = tf.reshape(c1, [batch_size, -1])

	# print('====c1 shape:', c1.get_shape())
	with tf.name_scope('Fully_connected1'):
		wfc1 = tf.get_variable(name='wfc1', shape=[c1.get_shape()[-1], 128], initializer=tf.initializers.truncated_normal(-0.1, 0.1))
		bfc1 = tf.get_variable(name='bfc1', shape=[128], initializer=tf.zeros_initializer())

		fc1 = tf.nn.leaky_relu(tf.add(tf.matmul(c1, wfc1), bfc1))
		fc1 = tf.nn.dropout(fc1, keep_prob)

	with tf.name_scope('prediction'):
		wp = tf.get_variable(name='wp', shape=[fc1.get_shape()[-1], 10], initializer=tf.initializers.truncated_normal(-0.1, 0.1))
		bp = tf.get_variable(name='bp', shape=[10], initializer=tf.zeros_initializer())

		prediction = tf.add(tf.matmul(fc1, wp), bp)

	print("Here are the trainable variables ...")
	print(*tf.trainable_variables(), sep='\n')

	with tf.name_scope('cost'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=prediction)) + \
		tf.reduce_mean([0.001*tf.nn.l2_loss(t) for t in tf.trainable_variables() if t.name.startswith('w')])

		tf.summary.scalar('cost', cost)

	with tf.name_scope('accuracy'):
		corr_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(prediction, 1))
		accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

		tf.summary.scalar('accuracy', accuracy)

	with tf.name_scope('train'):
		optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


	merged = tf.summary.merge_all()

	export_nodes = ['inputs', 'labels', 'keep_prob', 'prediction', 'accuracy', 'cost', 'optimizer', 'merged']

	Graph = namedtuple('Graph', export_nodes)
	local_dict = locals()
	graph = Graph(*[local_dict[each] for each in export_nodes])
    
	return graph



no_of_tr_batches = 5
no_of_ts_batches = 1



def train(model, epochs, log_string):
    '''Train the CNN'''

    saver = tf.train.Saver()
    
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        valid_loss_summary = []
        
        # Keep track of which batch iteration is being trained
        iteration = 0

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('../logs/t_{}'.format(log_string), sess.graph)
        valid_writer = tf.summary.FileWriter('../logs/v_{}'.format(log_string))

        for e in range(epochs):
            
            # Record progress with each epoch
            train_loss = []
            train_acc = []
            val_acc = []
            val_loss = []
            
            for i in tqdm(range(no_of_tr_batches), total=no_of_tr_batches):
                x, y = get_train_batch(i)
                feed = {model.inputs: x,
                        model.labels: sess.run(onehot(y)),
                        model.keep_prob: 0.6}
                summary, loss, acc, _ = sess.run([model.merged, 
                                                            model.cost, 
                                                            model.accuracy, 
                                                            model.optimizer], 
                                                        feed_dict=feed)                
                
                # Record the loss and accuracy of each training batch
                train_loss.append(loss)
                train_acc.append(acc)
                
                # Record the progress of training
                train_writer.add_summary(summary, iteration)
                
                iteration += 1
            
            # Average the training loss and accuracy of each epoch
            avg_train_loss = np.mean(train_loss)
            avg_train_acc = np.mean(train_acc) 


            for i in tqdm(range(no_of_ts_batches), total=no_of_ts_batches):
                x, y = get_train_batch(-1)
                feed = {model.inputs: x,
                        model.labels: sess.run(onehot(y)),
                        model.keep_prob: 1}
                summary, batch_loss, batch_acc, = sess.run([model.merged, 
                                                            model.cost, 
                                                            model.accuracy], 
                                                            feed_dict=feed)
                
                # Record the validation loss and accuracy of each epoch
                val_loss.append(batch_loss)
                val_acc.append(batch_acc)
            
            # Average the validation loss and accuracy of each epoch
            avg_valid_loss = np.mean(val_loss)    
            avg_valid_acc = np.mean(val_acc)
            valid_loss_summary.append(avg_valid_loss)
            
            # Record the validation data's progress
            valid_writer.add_summary(summary, iteration)

            # Print the progress of each epoch
            print("Epoch: {}/{}".format(e+1, epochs),
                  "Train Loss: {:.3f}".format(avg_train_loss),
                  "Train Acc: {:.3f}".format(avg_train_acc),
                  "Valid Loss: {:.3f}".format(avg_valid_loss),
                  "Valid Acc: {:.3f}".format(avg_valid_acc))

            # Stop training if the validation loss does not decrease after 3 epochs
            if avg_valid_loss > min(valid_loss_summary):
                print("No Improvement.")
                stop_early += 1
                if stop_early == 10:
                    break   
            
            # Reset stop_early if the validation loss finds a new low
            # Save a checkpoint of the model
            else:
                print("New Record!")
                stop_early = 0
                checkpoint = "../models/sentiment_{}.ckpt".format(log_string)
                saver.save(sess, checkpoint)

[_, unique] = ("%0.4f" %(time.time() / 1000)).split('.')
log_string = "basic-{}-{}".format(sys.argv[1], unique)
model = build_architecture()
train(model, 10000, log_string)

















