import os.path, os
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version, required TF 1.0
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Get the log directory
LOG_DIR = os.getcwd()+ "/logs"

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


    def load_vgg(sess, vgg_path):
        """
        Load Pretrained VGG Model into TensorFlow.
        :param sess: TensorFlow Session
        :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
        :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
        """

        # TODO: Implement function
        #   Use tf.saved_model.loader.load to load the model and weights
        vgg_tag = 'vgg16'
        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

        with tf.name_scope(vgg_tag):
            vgg_input_tensor_name = 'image_input:0'
            vgg_keep_prob_tensor_name = 'keep_prob:0'
            vgg_layer3_out_tensor_name = 'layer3_out:0'
            vgg_layer4_out_tensor_name = 'layer4_out:0'
            vgg_layer7_out_tensor_name = 'layer7_out:0'

            graph = tf.get_default_graph()

            input_img = graph.get_tensor_by_name(vgg_input_tensor_name)
            keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
            l3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
            l4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
            l7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

            tf.summary.histogram('layer4_out', l4_out)
            tf.summary.histogram('layer7_out', l7_out)


        return input_img, keep_prob, l3_out, l4_out, l7_out
    tests.test_load_vgg(load_vgg, tf)


    def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
        """
        Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
        :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
        :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
        :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
        :param num_classes: Number of classes to classify
        :return: The Tensor for the last layer of output
        """
        regular_val = 0.001

        with tf.name_scope("Layers"):
            l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,1,  padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))
            l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))
            l3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))

        with tf.name_scope("Up7_Skip4"):
            #upsample layer 7
            l7_upsampled = tf.layers.conv2d_transpose(l7_conv_1x1, num_classes, 4, 2, padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))
            # skip layer from layer 7 to layer 4
            l4_skip = tf.add(l7_upsampled,l4_conv_1x1)
            #upsample layer 4skip
        with tf.name_scope("Up4_Skip3"):
            l4_upsampled = tf.layers.conv2d_transpose(l4_skip, num_classes, 4, 2, padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))
            # skip layer from upsampled layer 4 to layer 3
            l3_skip = tf.add(l4_upsampled,l3_conv_1x1)
        with tf.name_scope("FCN_Output"):
            #upsample last
            output = tf.layers.conv2d_transpose(l3_skip, num_classes, 16, 8, padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_val))
        return output
    tests.test_layers(layers)


    def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """
        # TODO: Implement function
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
        training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy_loss)

        return logits, training_operation, cross_entropy_loss
    tests.test_optimize(optimize)


    def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
       correct_label, keep_prob, learning_rate):
        """
        Train neural network and print out the loss during training.
        :param sess: TF Session
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
        :param train_op: TF Operation to train the neural network
        :param cross_entropy_loss: TF Tensor for the amount of loss
        :param input_image: TF Placeholder for input images
        :param correct_label: TF Placeholder for label images
        :param keep_prob: TF Placeholder for dropout keep probability
        :param learning_rate: TF Placeholder for learning rate
        """
        # TODO Implement function

        # Hyperparameters
        lr = 0.001     # learning_rate
        kp = 0.5 # keep_prob

        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_DIR +"/model"+str(lr)+"_"+str(kp))
        writer.add_graph(sess.graph)

        # Hyperparameters
        lr = 0.001     # learning_rate
        kp = 0.5 # keep_prob
        print("Training NN ....")
        count = 1
        for i in range(epochs):
            for imgs, corr_labels in get_batches_fn(batch_size):
                if count % 20 == 0:
                    # Performing optimizer and loss
                    _, loss,s = sess.run([train_op, cross_entropy_loss,summ],
                                       feed_dict={input_image: imgs,
                                                  correct_label: corr_labels,
                                                  keep_prob: kp,
                                                  learning_rate: lr})
                    print("loss", loss, " epoch_ind ", i, " epochs ", epochs)
                    writer.add_summary(s, counter)

                else:
                    _, loss = sess.run([train_op, cross_entropy_loss],
                                          feed_dict={input_image: imgs,
                                                     correct_label: corr_labels,
                                                     keep_prob: kp,
                                                     learning_rate: lr})
                count += 1

    tests.test_train_nn(train_nn)


    def run():
        num_classes = 2
        image_shape = (160, 576)
        data_dir = './data'
        runs_dir = './runs'
        tests.test_for_kitti_dataset(data_dir)


        # Download pretrained vgg model
        helper.maybe_download_pretrained_vgg(data_dir)

        if os.path.exists(LOG_DIR+"/model*"):
            os.remove(LOG_DIR+"/model*")

        # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
        # You'll need a GPU with at least 10 teraFLOPS to train on.
        #  https://www.cityscapes-dataset.com/
        #Activate GPU
        config  = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'

        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, shape = (None, None, None, num_classes))
        epochs = 10
        batch_size = 16

        with tf.Session(config=config) as sess:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
            nn_final_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)



            logits, train_operation, cross_entropy_loss = optimize(nn_final_layer, correct_label, learning_rate, num_classes)
            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_operation, cross_entropy_loss, input_image, correct_label,
                keep_prob, learning_rate)

            #Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

            # OPTIONAL: Apply the trained model to a video


    if __name__ == '__main__':
                run()
