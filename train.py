#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import timedelta
from importlib import import_module
import logging.config
import os
from signal import SIGINT, SIGTERM
import sys
import time

import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import matplotlib

matplotlib.use('Agg')

import common
import lbtoolbox as lb
from nets import NET_CHOICES
from heads import HEAD_CHOICES
from ranking import LOSS_CHOICES,METRIC_CHOICES
from ranking.semi_hard_triplet import triplet_semihard_loss
from ranking.lifted_structured import lifted_loss
from ranking.npair import npairs_loss
from ranking.angular import angular_loss

parser = ArgumentParser(description='Train a ReID network.')

# Required.

parser.add_argument(
    '--experiment_root', required=True, type=common.writeable_directory,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--train_set',
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

# Optional with sane defaults.

parser.add_argument(
    '--resume', action='store_true', default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

parser.add_argument(
    '--model_name', default='resnet_v1_50', choices=NET_CHOICES,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default='fc1024', choices=HEAD_CHOICES,
    help='Name of the head to use.')

parser.add_argument(
    '--embedding_dim', default=128, type=common.positive_int,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=None,
    help='Path to the checkpoint file of the pretrained network.')

# TODO move these defaults to the .sh script?
parser.add_argument(
    '--batch_p', default=32, type=common.positive_int,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--batch_k', default=4, type=common.positive_int,
    help='The numberK used in the PK-batches')

parser.add_argument(
    '--net_input_height', default=256, type=common.positive_int,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=128, type=common.positive_int,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--pre_crop_height', default=288, type=common.positive_int,
    help='Height used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--pre_crop_width', default=144, type=common.positive_int,
    help='Width used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')
# TODO end

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--margin', default='soft', type=common.float_or_string,
    help='What margin to use: a float value for hard-margin, "soft" for '
         'soft-margin, or no margin if "none".')

parser.add_argument(
    '--metric', default='euclidean', choices=METRIC_CHOICES,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--loss', default='batch_hard', choices=LOSS_CHOICES,
    help='Enable the super-mega-advanced top-secret sampling stabilizer.')

parser.add_argument(
    '--learning_rate', default=3e-4, type=common.positive_float,
    help='The initial value of the learning-rate, before it kicks in.')

parser.add_argument(
    '--train_iterations', default=25000, type=common.positive_int,
    help='Number of training iterations.')

parser.add_argument(
    '--decay_start_iteration', default=15000, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument(
    '--gpu', default='0', type=str,
    help='Which GPU to use')

parser.add_argument(
    '--checkpoint_frequency', default=1000, type=common.nonnegative_int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', action='store_true', default=False,
    help='When this flag is provided, crop augmentation is performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')

parser.add_argument(
    '--detailed_logs', action='store_true', default=False,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')

parser.add_argument(
    '--augment', action='store_true', default=False, help='Data augmentation with imgaug')


def sample_k_fids_for_pid(pid, all_fids, all_pids, batch_k):
    """ Given a PID, select K FIDs of that specific PID. """
    possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

    # The following simply uses a subset of K of the possible FIDs
    # if more than, or exactly K are available. Otherwise, we first
    # create a padded list of indices which contain a multiple of the
    # original FID count such that all of them will be sampled equally likely.
    count = tf.shape(possible_fids)[0]
    padded_count = tf.cast(tf.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
    full_range = tf.mod(tf.range(padded_count), count)

    # Sampling is always performed by shuffling and taking the first k.
    shuffled = tf.random_shuffle(full_range)
    selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

    return selected_fids, tf.fill([batch_k], pid)




def main(argv):
    args = parser.parse_args(argv)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # We store all arguments in a json file. This has two advantages:
    # 1. We can always get back and see what exactly that experiment was
    # 2. We can resume an experiment as-is without needing to remember all flags.
    args_file = os.path.join(args.experiment_root, 'args.json')
    if args.resume:
        if not os.path.isfile(args_file):
            raise IOError('`args.json` not found in {}'.format(args_file))

        print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)
        args_resumed['resume'] = True  # This would be overwritten.

        # When resuming, we not only want to populate the args object with the
        # values from the file, but we also want to check for some possible
        # conflicts between loaded and given arguments.
        for key, value in args.__dict__.items():
            if key in args_resumed:
                resumed_value = args_resumed[key]
                if resumed_value != value:
                    print('Warning: For the argument `{}` we are using the'
                          ' loaded value `{}`. The provided value was `{}`'
                          '.'.format(key, resumed_value, value))
                    args.__dict__[key] = resumed_value
            else:
                print('Warning: A new argument was added since the last run:'
                      ' `{}`. Using the new value: `{}`.'.format(key, value))

    else:
        # If the experiment directory exists already, we bail in fear.
        if os.path.exists(args.experiment_root):
            if os.listdir(args.experiment_root):
                print('The directory {} already exists and is not empty.'
                      ' If you want to resume training, append --resume to'
                      ' your call.'.format(args.experiment_root))
                exit(1)
        else:
            os.makedirs(args.experiment_root)

        # Store the passed arguments for later resuming and grepping in a nice
        # and readable format.
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

    log_file = os.path.join(args.experiment_root, "train")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('train')

    # Also show all parameter values at the start, for ease of reading logs.
    log.info('Training using the following parameters:')
    for key, value in sorted(vars(args).items()):
        log.info('{}: {}'.format(key, value))

    # Check them here, so they are not required when --resume-ing.
    if not args.train_set:
        parser.print_help()
        log.error("You did not specify the `train_set` argument!")
        sys.exit(1)
    if not args.image_root:
        parser.print_help()
        log.error("You did not specify the required `image_root` argument!")
        sys.exit(1)

    # Load the data from the CSV file.
    pids, fids = common.load_dataset(args.train_set, args.image_root)
    max_fid_len = max(map(len, fids))  # We'll need this later for logfiles.

    # Setup a tf.Dataset where one "epoch" loops over all PIDS.
    # PIDS are shuffled after every epoch and continue indefinitely.
    unique_pids = np.unique(pids)
    if len(unique_pids) < args.batch_p:
        unique_pids = np.tile(unique_pids, int(np.ceil(args.batch_p / len(unique_pids))))
    dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
    dataset = dataset.shuffle(len(unique_pids))

    # Constrain the dataset size to a multiple of the batch-size, so that
    # we don't get overlap at the end of each epoch.
    dataset = dataset.take((len(unique_pids) // args.batch_p) * args.batch_p)
    dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.

    # For every PID, get K images.
    dataset = dataset.map(lambda pid: sample_k_fids_for_pid(
        pid, all_fids=fids, all_pids=pids, batch_k=args.batch_k))

    # Ungroup/flatten the batches for easy loading of the files.
    dataset = dataset.apply(tf.contrib.data.unbatch())

    # Convert filenames to actual image tensors.
    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    dataset = dataset.map(
        lambda fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)


    # Augment the data if specified by the arguments.

    dataset = dataset.map(
        lambda im, fid, pid: common.fid_to_image(
            fid, pid, image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),  # Ergys
        num_parallel_calls=args.loading_threads)

    if args.flip_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.image.random_flip_left_right(im), fid, pid))
    if args.crop_augment:
        dataset = dataset.map(
            lambda im, fid, pid: (tf.random_crop(im, net_input_size + (3,)), fid, pid))

    # Group it back into PK batches.
    batch_size = args.batch_p * args.batch_k
    dataset = dataset.batch(batch_size)

    # Overlap producing and consuming for parallelism.
    dataset = dataset.prefetch(1)

    # Since we repeat the data infinitely, we only need a one-shot iterator.
    images, fids, pids = dataset.make_one_shot_iterator().get_next()

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.

    weight_decay = 10e-4
    weights_regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    endpoints, body_prefix = model.endpoints(images, is_training=True)
    with tf.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=True,weights_regularizer=weights_regularizer)

    # Create the loss in two steps:
    # 1. Compute all pairwise distances according to the specified metric.
    # 2. For each anchor along the first dimension, compute its loss.
    # batch_embedding = endpoints['emb']
    batch_embedding = endpoints['emb']
    if args.loss == 'semi_hard_triplet':
        triplet_loss = triplet_semihard_loss(batch_embedding, pids, args.margin,args.metric)
    elif args.loss == 'lifted_loss':
        triplet_loss = lifted_loss(pids, batch_embedding, margin=args.margin)
    # elif args.loss == 'contrastive_loss':
    #     # embeddings_anchor, embeddings_positive = tf.unstack(tf.reshape(batch_embedding, [-1, 2, args.embedding_dim]), 2,1)
    #     # anchor_ids, pos_ids = tf.unstack(tf.reshape(pids, [-1, 2, 1]), 2, 1)
    #     assert batch_size % 2 == 0
    #     assert args.batch_k == 4  ## Can work with other number but will need tuning
    #
    #     contrastive_idx = np.tile([0, 1, 4, 3, 2, 5, 6, 7], args.batch_p // 2)
    #     for i in range(args.batch_p // 2):
    #         contrastive_idx[i * 8:i * 8 + 8] += i * 8
    #
    #     contrastive_idx = np.expand_dims(contrastive_idx, 1)
    #     batch_embedding_ordered = tf.gather_nd(batch_embedding, contrastive_idx)
    #     pids_ordered = tf.gather_nd(pids, contrastive_idx)
    #     # batch_embedding_ordered = tf.Print(batch_embedding_ordered,[pids_ordered],'pids_ordered :: ',summarize=1000)
    #     embeddings_anchor, embeddings_positive = tf.unstack(
    #         tf.reshape(batch_embedding_ordered, [-1, 2, args.embedding_dim]), 2,
    #         1)
    #     # embeddings_anchor = tf.Print(embeddings_anchor,[pids_ordered,embeddings_anchor,embeddings_positive,batch_embedding,batch_embedding_ordered],"Tensors ", summarize=1000)
    #
    #     fixed_labels = np.tile([1, 0, 0, 1], args.batch_p // 2)
    #     # fixed_labels = np.reshape(fixed_labels,(len(fixed_labels),1))
    #     # print(fixed_labels)
    #     labels = tf.constant(fixed_labels)
    #     # labels = tf.Print(labels,[labels],'labels ',summarize=1000)
    #     triplet_loss = loss.LOSS_CHOICES[args.loss](labels, embeddings_anchor, embeddings_positive,
    #                                                 margin=args.margin, log_var=None)
    # elif args.loss == 'lifted_loss' or args.loss == 'soft_lifted_loss':
    #     triplet_loss = loss.LOSS_CHOICES[args.loss](pids, batch_embedding, margin=args.margin,
    #                                                 log_var=endpoints['data_sigma'])
    #
    elif args.loss == 'angular_loss':
        embeddings_anchor, embeddings_positive = tf.unstack(tf.reshape(batch_embedding, [-1, 2, args.embedding_dim]), 2,
                                                            1)
        # pids = tf.Print(pids, [pids], 'pids:: ', summarize=100)
        pids, _ = tf.unstack(tf.reshape(pids, [-1, 2, 1]), 2, 1)
        # pids = tf.Print(pids,[pids],'pids:: ',summarize=100)
        triplet_loss = angular_loss(pids, embeddings_anchor, embeddings_positive,
                                                    batch_size=args.batch_p, with_l2reg=True)
    elif args.loss == 'npairs_loss':
         assert args.batch_k == 2  ## Single positive pair per class
         embeddings_anchor, embeddings_positive = tf.unstack(tf.reshape(batch_embedding, [-1, 2, args.embedding_dim]), 2,                                                         1)
         pids, _ = tf.unstack(tf.reshape(pids, [-1, 2, 1]), 2, 1)
         pids = tf.reshape(pids, [-1])
         triplet_loss = npairs_loss(pids, embeddings_anchor, embeddings_positive)
    else:
        raise NotImplementedError('loss function {} NotImplemented'.format(args.loss))


    loss_mean = tf.reduce_mean(triplet_loss)


    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)

    # Define the optimizer and the learning-rate schedule.
    # Unfortunately, we get NaNs if we don't handle no-decay separately.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if 0 <= args.decay_start_iteration < args.train_iterations:
        learning_rate = tf.train.polynomial_decay(args.learning_rate, global_step, args.train_iterations,
                                                  end_learning_rate=1e-7, power=1)
    else:
        learning_rate = args.learning_rate

    optimizer = tf.train.AdamOptimizer(learning_rate)

    #
    # learning_rate = tf.train.polynomial_decay(args.learning_rate, global_step,
    #                                           args.train_iterations, end_learning_rate= 1e-7,
    #                                           power=1)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    # Feel free to try others!
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate)

    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss_mean, global_step=global_step)

    # Define a saver for the complete model.
    checkpoint_saver = tf.train.Saver(max_to_keep=2)
    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=gpu_config) as sess:
        if args.resume:
            # In case we're resuming, simply load the full checkpoint to init.
            last_checkpoint = tf.train.latest_checkpoint(args.experiment_root)
            if last_checkpoint == None:
                print('Resume with No previous checkpoint')

                # But if we're starting from scratch, we may need to load some
                # variables from the pre-trained weights, and random init others.
                sess.run(tf.global_variables_initializer())
                if args.initial_checkpoint is not None:
                    saver = tf.train.Saver(model_variables)
                    saver.restore(sess, args.initial_checkpoint)

                # In any case, we also store this initialization as a checkpoint,
                # such that we could run exactly reproduceable experiments.
                checkpoint_saver.save(sess, os.path.join(
                    args.experiment_root, 'checkpoint'), global_step=0)

            else:
                log.info('Restoring from checkpoint: {}'.format(last_checkpoint))
                checkpoint_saver.restore(sess, last_checkpoint)
        else:
            # But if we're starting from scratch, we may need to load some
            # variables from the pre-trained weights, and random init others.
            sess.run(tf.global_variables_initializer())
            if args.initial_checkpoint is not None:
                saver = tf.train.Saver(model_variables)
                saver.restore(sess, args.initial_checkpoint)

            # In any case, we also store this initialization as a checkpoint,
            # such that we could run exactly reproduceable experiments.
            checkpoint_saver.save(sess, os.path.join(
                args.experiment_root, 'checkpoint'), global_step=0)

        start_step = sess.run(global_step)
        log.info('Starting training from iteration {}.'.format(start_step))

        # Finally, here comes the main-loop. This `Uninterrupt` is a handy
        # utility such that an iteration still finishes on Ctrl+C and we can
        # stop the training cleanly.
        with lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:
            for i in range(start_step, args.train_iterations):

                # Compute gradients, update weights, store logs!
                start_time = time.time()
                _,  step, b_embs, b_loss, b_fids = \
                    sess.run([train_op, global_step, endpoints['emb'], triplet_loss, fids])
                elapsed_time = time.time() - start_time



                # Do a huge print out of the current progress.
                seconds_todo = (args.train_iterations - step) * elapsed_time
                log.info('iter:{:6d}, loss min|avg|max: {:.3f}|{:.3f}|{:6.3f}, ETA: {} ({:.2f}s/it)'.format(
                    step,
                    float(np.min(b_loss)),
                    float(np.mean(b_loss)),
                    float(np.max(b_loss)),
                    # args.batch_k - 1, float(b_prec_at_k),
                    timedelta(seconds=int(seconds_todo)),
                    elapsed_time))
                sys.stdout.flush()
                sys.stderr.flush()

                # Save a checkpoint of training every so often.
                if (args.checkpoint_frequency > 0 and
                                step % args.checkpoint_frequency == 0):
                    checkpoint_saver.save(sess, os.path.join(
                        args.experiment_root, 'checkpoint'), global_step=step)

                # Stop the main-loop at the end of the step, if requested.
                if u.interrupted:
                    log.info("Interrupted on request!")
                    break

        # Store one final checkpoint. This might be redundant, but it is crucial
        # in case intermediate storing was disabled and it saves a checkpoint
        # when the process was interrupted.
        checkpoint_saver.save(sess, os.path.join(
            args.experiment_root, 'checkpoint'), global_step=step)



if __name__ == '__main__':

    import getpass
    username = getpass.getuser()
    if username == 'ahmedtaha':

        dataset_dir = '/Users/ahmedtaha/Documents/dataset/'
        # dataset_dir = '/Volumes/Backup/datasets/'
        trained_models_dir = '/Users/ahmedtaha/Documents/Model/'
        experiment_root_dir = '/Users/ahmedtaha/Documents/dataset/retrieval_baseline/'
        local = True
    elif username == 'ataha':

        dataset_dir = '/mnt/work/datasets/'
        trained_models_dir = '/mnt/work/datasets/Model/'
        experiment_root_dir = dataset_dir + 'Market-1501-v15.09.15/experiment/'

        local = False
    elif username == 'ahmdtaha':

        # dataset_dir = '/scratch0/ahmdtaha/'
        dataset_dir = '/vulcan/scratch/ahmdtaha/'
        trained_models_dir = '/vulcan/scratch/ahmdtaha/weights/'
        experiment_root_dir = '/vulcan/scratch/ahmdtaha/' + 'Market-1501-v15.09.15/experiment/'

        local = False
    else:
        raise NotImplementedError('User {} not found'.format(username))


    dataset_name = 'inshop'
    if dataset_name == 'cub':
        db_dir = 'CUB_200_2011/images'
        train_file = 'cub_train.csv'
        extra_args = [
            '--batch_p', '10',
            '--batch_k', '6',
            '--flip_augment',
            '--crop_augment',

        ]
    elif dataset_name == 'inshop':
        db_dir = 'In_shop_Clothes_Retrieval_Benchmark'
        train_file = 'deep_fashion_train.csv'
        extra_args = [
            # p_10,k_6
            '--batch_p', '30',
            '--batch_k', '2',
            '--net_input_height', '224',
            '--net_input_width', '224',
            '--pre_crop_height', '256',
            '--pre_crop_width', '256',
            '--flip_augment',
            '--crop_augment',
        ]
    else:
        raise NotImplementedError('invalid dataset {}'.format(dataset_name))

    arg_loss = 'angular_loss'
    arg_head = 'fc1024'
    arg_margin = '1.0'

    exp_dir = [dataset_name,arg_head,arg_loss,'m_{}'.format(arg_margin),'densenet']
    exp_dir = '_'.join(exp_dir)

    args = [
        '--image_root', dataset_dir + db_dir,
        '--experiment_root', experiment_root_dir + exp_dir,

        #'--initial_checkpoint', trained_models_dir + 'resnet_v1_50/resnet_v1_50.ckpt',
        #'--model_name', 'resnet_v1_50',

        '--initial_checkpoint', trained_models_dir + 'tf-densenet169/tf-densenet169.ckpt',
        '--model_name', 'densenet169',

        '--train_set', './data/' + train_file,

        '--net_input_height', '224',
        '--net_input_width', '224',
        '--pre_crop_height', '256',
        '--pre_crop_width', '256',

        # '--resume',
        '--head_name', arg_head,
        '--margin', arg_margin,
        '--loss', arg_loss,
        '--gpu', '1',
    ]

    args.extend(extra_args)

    main(args)

