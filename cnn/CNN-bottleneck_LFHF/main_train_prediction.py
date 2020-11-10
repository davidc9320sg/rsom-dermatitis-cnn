""" this code is to save the model at the epoch that is not saved previously. 
user will input DESIRED_VAL_ACC, model_checkpoint. this code will load the model at
the checkpoint and train untill obtained the DESIRED_VAL_ACC"""

import numpy as np
import tensorflow as tf
import random as rn
import os
from Build_Model import CNN
from Batch_train_valid import batch_train_valid
from loss_function import Loss
from optimizer import opt
import time


if __name__ == '__main__':
    experiment_type = 'mild_vs_modsev' #'healthy_vs_disease'
    checkpoint_epoch = None

    # set seed
    seed = 34526
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # set parameters
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    EPOCHS = 50
    epochs_to_decay = 2
    channels = 2  # [FIXED - LF HF]
    CNN_name = "rsom"

    if experiment_type == 'healthy_vs_disease' or experiment_type == 'mild_vs_modsev':
        prediction_output = 2
    elif experiment_type == 'three_severities':
        prediction_output = 3
    
    for cv in [1]:
        print('CV {}'.format(cv))
        folder_to_read = '../../data/CV{}/{}'.format(cv, experiment_type)
        output_folder = 'results/CV{}/{}/'.format(cv, experiment_type)
        if os.path.exists(output_folder) is False:
            os.makedirs(output_folder)
        if checkpoint_epoch is not None:
            checkpoint_dir = 'results/CV{}/{}/model/my_model.ckpt-{}'.format(cv, experiment_type, checkpoint_epoch)
        else: 
            checkpoint_dir = None
               
        # load batch data
        load_batch_data = batch_train_valid(experiment_type, folder_to_read, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE)
        n_batch_train = load_batch_data.n_train_batch
        n_batch_valid = load_batch_data.n_valid_batch
        batch_train_arr, batch_train_label_arr = load_batch_data.batch_train()
        batch_valid_arr, batch_valid_label_arr = load_batch_data.batch_valid()
        width = load_batch_data.batch_train_arr.shape[2]
        height = load_batch_data.batch_train_arr.shape[3]
        depth = load_batch_data.batch_train_arr.shape[4]
        starter_learning_rate = 0.00001
    
        tf.reset_default_graph()
         
        # create placeholder
        X = tf.placeholder(tf.float32, shape=[None, channels, width, height, depth],name='X')
        Y = tf.placeholder(tf.float32, shape=[None, prediction_output], name='Y')
        #features = tf.placeholder(tf.float32, shape=[None, n_features], name='ft')

        global_step = tf.Variable(0, trainable=False, name='global_step')
        decay_step = n_batch_train * epochs_to_decay
        # learning rate decay every 2 epochs
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                  decay_step, 0.96, staircase=True)

        my_cnn = CNN(CNN_name, prediction_output)
        my_out = my_cnn._build_model(X)

        cross_entropy = Loss(Y, my_out).ce
        loss_op = tf.reduce_mean(cross_entropy)
        optimizer = opt(learning_rate).Adam
        train_op = optimizer.minimize(loss_op,global_step=global_step)

        comp_pred = tf.equal( tf.argmax(Y, 1), tf.argmax(my_out, 1) )
        accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))
        # Add an op to initialize the variables
        init = tf.global_variables_initializer()

        model_saver = tf.train.Saver(max_to_keep=200)
        text = ''

        with tf.Session() as sess:
            if checkpoint_dir is not None:
                model_saver.restore(sess, checkpoint_dir)
                print("Model restored")
                # define start and end epoch value
                start_epoch, end_epoch = checkpoint_epoch, checkpoint_epoch + EPOCH
            else:
                # Actually initialize the variables
                sess.run(init)
                start_epoch, end_epoch = 0, EPOCHS

            train_acc = []
            train_loss = []
            valid_acc = []
            valid_loss = []

            # avg loss : load all data and sum the loss each epoch and then average the loss

            for epoch in range(start_epoch, end_epoch):

                epoch_acc = 0
                epoch_loss = 0
                start = time.time()

                # TRAINING NETWORK
                lr = sess.run(optimizer._lr)
                for i in range(n_batch_train):

                    #print("batch_train start ", i)
                    # avg gradient
                    train_data = load_batch_data.batch_train()
                 
                    _, loss, acc = sess.run([train_op,loss_op,accuracy],feed_dict={X: train_data[0], Y: train_data[1]})

                    epoch_acc += acc
                    epoch_loss += loss
                    #print("batch train end")

                epoch_train_acc = epoch_acc / n_batch_train
                epoch_train_loss = epoch_loss / n_batch_train

                #print('=====================================')

                epoch_v_acc = 0
                epoch_v_loss = 0

                for i in range(n_batch_valid):
                    #print("batch valid start ", i)
                    valid_data = load_batch_data.batch_valid()
                    val_loss, val_acc = sess.run([loss_op,accuracy], feed_dict={X: valid_data[0], Y: valid_data[1]})

                    epoch_v_acc += val_acc
                    epoch_v_loss += val_loss
                    #print("batch valid end")

                epoch_val_acc = epoch_v_acc / n_batch_valid
                epoch_val_loss = epoch_v_loss / n_batch_valid
                end = time.time()
 

                print("lr:{:.2e}, epoch:{}, Acc:{:.4f}, Loss:{:.4f}, Acc_val:{:.4f}, Loss_val:{:.4f}, time:{:.2f}min".format(lr, epoch, epoch_train_acc,epoch_train_loss,epoch_val_acc, epoch_val_loss, (end-start)/60))
                text = text + str(epoch) + ' ' + str(epoch_train_acc) + ' ' + str(epoch_train_loss) + ' ' + str(epoch_val_acc) + ' ' + str(epoch_val_loss) + '\n'
                #print('===============================')

                train_acc.append(epoch_train_acc)
                train_loss.append(epoch_train_loss)
                valid_acc.append(epoch_val_acc)
                valid_loss.append(epoch_val_loss)
               
                """ 
                if epoch % 1 == 0:
                    with open(output_folder + '/loss_curve.txt', 'w') as fp:
                        fp.write(text)
                    fp.close()
                """

                # Save the variables to disk
                save_path = model_saver.save(sess, output_folder + "/model/my_model.ckpt",global_step = epoch)
                print("Model saved in path: %s" % save_path)
        del load_batch_data, batch_train_arr, batch_train_label_arr, batch_valid_arr, batch_valid_label_arr, train_data, valid_data                