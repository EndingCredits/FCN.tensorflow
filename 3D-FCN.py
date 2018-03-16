from __future__ import print_function
import tensorflow as tf
import numpy as np

from glob import glob
import os

import TensorflowUtils as utils
import datetime
import LungDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs3D/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "./lung_data", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo3D/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSES = 3+1
IMAGE_SIZE = 64
IMAGE_DEPTH = 10

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    
    #using architecture described in https://arxiv.org/pdf/1704.06382.pdf
    #TODO: Look up preprocessing step
    processed_image = image/100

    with tf.variable_scope("inference"):
    
        def down_layer(x, in_size, out_size, i):
            # Down 1
            W1 = utils.weight_variable([3, 3, 3, in_size, out_size//2], name="W_d_"+str(i)+"_1")
            b1 = utils.bias_variable([out_size//2], name="b_d_"+str(i)+"_1")
            conv1 = utils.conv3d_basic(x, W1, b1)
            relu1 = tf.nn.relu(conv1, name="relu_d_"+str(i)+"_1")
            
            # Down 2
            W2 = utils.weight_variable([3, 3, 3, out_size//2, out_size], name="W_d_"+str(i)+"_2")
            b2 = utils.bias_variable([out_size], name="b_d_"+str(i)+"_2")
            conv2 = utils.conv3d_basic(relu1, W2, b2)
            relu2 = tf.nn.relu(conv2, name="relu_d_"+str(i)+"_2")
            
            # Pool 1
            pool = utils.max_pool_2x2x2(relu2)
            
            return relu2, pool
            
            
        # Apply 4 down layes of increasing sizes
        d1, p1 = down_layer(processed_image, 1, 64, 1)
        d2, p2 = down_layer(p1, 64, 128, 2)
        d3, p3 = down_layer(p2, 128, 256, 3)
        d4, p4 = down_layer(p3, 256, 512, 4)
        
        
        def up_layer(x1, x2, in_size1, in_size2, out_size, i):
            # Up 1
            W1 = utils.weight_variable([2, 2, 2, in_size2, in_size1], name="W_u_"+str(i)+"_1")
            b1 = utils.bias_variable([in_size2], name="b_u_"+str(i)+"_1")
            deco1 = utils.conv3d_transpose_strided(x1, W1, b1, output_shape=tf.shape(x2))
            relu1 = tf.nn.relu(deco1, name="relu_d_"+str(i)+"_1")
            
            # Concat
            conc1 = tf.concat([relu1, x2], -1) # concat along the channels dimension
            
            # Conv1
            W2 = utils.weight_variable([3, 3, 3, in_size2*2, out_size], name="W_u_"+str(i)+"_2")
            b2 = utils.bias_variable([out_size], name="b_u_"+str(i)+"_2")
            conv1 = utils.conv3d_basic(conc1, W2, b2)
            relu2 = tf.nn.relu(conv1, name="relu_u_"+str(i)+"_2")
            
            # Conv2
            W3 = utils.weight_variable([3, 3, 3, out_size, out_size], name="W_u_"+str(i)+"_3")
            b3 = utils.bias_variable([out_size], name="b_u_"+str(i)+"_3")
            conv3 = utils.conv3d_basic(relu2, W3, b3)
            relu4 = tf.nn.relu(conv3, name="relu_u_"+str(i)+"_3")
            
            return relu4
            
        #Apply 3 Up layers with skip connections
        u3 = up_layer(d4, d3, 512, 256, 256, 3)
        u2 = up_layer(u3, d2, 256, 128, 128, 2)
        u1 = up_layer(u2, d1, 128, 64, 64, 1)
        
        #Apply a final Conv layer
        W = utils.weight_variable([3, 3, 3, 64, NUM_OF_CLASSES], name="W_o")
        b = utils.bias_variable([NUM_OF_CLASSES], name="b_o")
        conv = utils.conv3d_basic(u1, W, b)
        
        annotation_pred = tf.argmax(conv, dimension=4, name="prediction")

    return tf.expand_dims(annotation_pred, dim=4), conv


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    # Set up TF placeholderd
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, 1], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, NUM_OF_CLASSES], name="annotation")
    
    # Build graph
    print("Building graph...")
    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image[..., IMAGE_DEPTH//2, :], max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation[..., IMAGE_DEPTH//2, 1:]*255, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", logits[..., IMAGE_DEPTH//2, 1:]*255, max_outputs=2)
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=annotation, name="entropy")))
    tf.summary.scalar("entropy", loss)

    # Add variables to summaries as needed
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up dataset readers")
    train_dataset_reader = dataset.LungDatset("LungTrainingData.mat", None)
    validation_dataset_reader = train_dataset_reader

    print("Setting up tensorflow session (you'll probably get a bunch of warnings now)")
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        print("%s Training..." % (datetime.datetime.now()))
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 1 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("%s Step: %d, Train_loss:%g" % (datetime.datetime.now().strftime('%H:%M:%S'), itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 10 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss,
                    feed_dict={image: valid_images,
                               annotation: valid_annotations,
                               keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now().strftime('%H:%M:%S'), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        print("Running visualisation...")
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
          for lay in range(IMAGE_DEPTH):
            utils.save_image(valid_images[itr,...,lay,:].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr)+"."+str(lay))
            utils.save_image(valid_annotations[itr,...,lay,:].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr)+"."+str(lay))
            utils.save_image(pred[itr,...,lay,:].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr)+"."+str(lay))
            print("Saved image: %d" % itr)

def colour_img(image):
    return image[:,:,1:]*255
    

if __name__ == "__main__":
    tf.app.run()
    
    
#TODO: Remove this once it's tested
"""
# Down 2_1
W2_1 = utils.weight_variable([3, 3, 3, 64, 64], name="W2_1")
b2_1 = utils.bias_variable([64], name="b2_1")
conv2_1 = utils.conv3d(pool_1, W2_1, b2_1)
relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")

# Down 2_2
W2_2 = utils.weight_variable([3, 3, 3, 64, 128], name="W2_2")
b2_2 = utils.bias_variable([128], name="b2_2")
conv2_2 = utils.conv3d_basic(relu2_1, W2_2, b2_2)
relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")

# Pool 2
pool_2 = utils.max_pool_2x2(relu2_2)


# Down 3_1
W3_1 = utils.weight_variable([3, 3, 3, 128, 129], name="W3_1")
b3_1 = utils.bias_variable([128], name="b3_1")
conv3_1 = utils.conv3d(pool_2, W3_1, b3_1)
relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")

# Down 2_2
W3_2 = utils.weight_variable([3, 3, 3, 128, 256], name="W3_2")
b3_2 = utils.bias_variable([256], name="b3_2")
conv3_2 = utils.conv3d_basic(relu3_1, W3_2, b3_2)
relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")

# Pool 2
pool_3 = utils.max_pool_2x2(relu3_2)


# Down 4_1
W4_1 = utils.weight_variable([3, 3, 3, 256, 256], name="W4_1")
b4_1 = utils.bias_variable([256], name="b4_1")
conv4_1 = utils.conv3d(pool_3, W4_1, b4_1)
relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")

# Down 4_2
W4_2 = utils.weight_variable([3, 3, 3, 256, 512], name="W4_2")
b4_2 = utils.bias_variable([512], name="b4_2")
conv4_2 = utils.conv3d_basic(relu4_1, W4_2, b4_2)
relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")
"""
