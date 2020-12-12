#!/usr/bin/env python3
import os
import h5py
import json
import psutil
import common
import numpy as np
import logging.config
import os.path as osp
import tensorflow as tf
import constants as const
from itertools import count
from utils import os_utils
from aggregators import AGGREGATORS
from argparse import ArgumentParser
from model.embedding_model import EmbeddingModel




parser = ArgumentParser(description='Embed a dataset using a trained network.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--dataset', required=True,
    help='Path to the dataset csv file to be embedded.')

# Optional

parser.add_argument(
    '--image_root', type=common.readable_directory,
    help='Path that will be pre-pended to the filenames in the train_set csv.')

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--loading_threads', default=8, type=common.positive_int,
    help='Number of threads used for parallel data loading.')



parser.add_argument(
    '--batch_size', default=64, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on available memory.')

parser.add_argument(
    '--filename', default=None,
    help='Name of the HDF5 file in which to store the embeddings, relative to'
         ' the `experiment_root` location. If omitted, appends `_embeddings.h5`'
         ' to the dataset name.')

parser.add_argument(
    '--foldername', default=None,
    help='Name of dir to save embeds')

parser.add_argument(
    '--flip_augment', action='store_true', default=False,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', choices=['center', 'avgpool', 'five'], default=None,
    help='When this flag is provided, crop augmentation is performed.'
         '`avgpool` means the full image at the precrop size is used and '
         'the augmentation is performed by the average pooling. `center` means'
         'only the center crop is used and `five` means the four corner and '
         'center crops are used. When not provided, by default the image is '
         'resized to network input size.')

parser.add_argument(
    '--aggregator', choices=AGGREGATORS.keys(), default=None,
    help='The type of aggregation used to combine the different embeddings '
         'after augmentation.')

parser.add_argument(
    '--quiet', action='store_true', default=False,
    help='Don\'t be so verbose.')


def flip_augment(image, fid, pid):
    """ Returns both the original and the horizontal flip of an image. """
    images = tf.stack([image, tf.reverse(image, [1])])
    return images, tf.stack([fid]*2), tf.stack([pid]*2)

def five_crops(image, crop_size):
    """ Returns the central and four corner crops of `crop_size` from `image`. """
    image_size = tf.shape(image)[:2]
    crop_margin = tf.subtract(image_size, crop_size)
    assert_size = tf.debugging.assert_non_negative(
        crop_margin, message='Crop size must be smaller or equal to the image size.')
    with tf.control_dependencies([assert_size]):
        top_left = tf.compat.v1.floor_div(crop_margin, 2)
        bottom_right = tf.math.add(top_left, crop_size)

    center       = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    top_left     = image[:-crop_margin[0], :-crop_margin[1]]
    top_right    = image[:-crop_margin[0], crop_margin[1]:]
    bottom_left  = image[crop_margin[0]:, :-crop_margin[1]]
    bottom_right = image[crop_margin[0]:, crop_margin[1]:]
    return center, top_left, top_right, bottom_left, bottom_right


def main(argv):
    # Verify that parameters are set correctly.
    args = parser.parse_args(argv)

    if not os.path.exists(args.dataset):
        return

    # Possibly auto-generate the output filename.
    if args.filename is None:
        basename = os.path.basename(args.dataset)
        args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'

    os_utils.touch_dir(os.path.join(args.experiment_root,args.foldername))

    log_file = os.path.join(args.experiment_root,args.foldername, "embed")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('embed')

    args.filename = os.path.join(args.experiment_root,args.foldername, args.filename)
    var_filepath = os.path.join(args.experiment_root, args.foldername, args.filename[:-3] + '_var.txt')
    # Load the args from the original experiment.
    args_file = os.path.join(args.experiment_root, 'args.json')

    if os.path.isfile(args_file):
        if not args.quiet:
            print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # Add arguments from training.
        for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

        # A couple special-cases and sanity checks
        if (args_resumed['crop_augment']) == (args.crop_augment is None):
            print('WARNING: crop augmentation differs between training and '
                  'evaluation.')
        args.image_root = args.image_root or args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))

    # Check a proper aggregator is provided if augmentation is used.
    if args.flip_augment or args.crop_augment == 'five':
        if args.aggregator is None:
            print('ERROR: Test time augmentation is performed but no aggregator'
                  'was specified.')
            exit(1)
    else:
        if args.aggregator is not None:
            print('ERROR: No test time augmentation that needs aggregating is '
                  'performed but an aggregator was specified.')
            exit(1)

    if not args.quiet:
        print('Evaluating using the following parameters:')
        for key, value in sorted(vars(args).items()):
            print('{}: {}'.format(key, value))

    # Load the data from the CSV file.
    _, data_fids = common.load_dataset(args.dataset, args.image_root)

    net_input_size = (args.net_input_height, args.net_input_width)
    pre_crop_size = (args.pre_crop_height, args.pre_crop_width)

    # Setup a tf Dataset containing all images.
    dataset = tf.data.Dataset.from_tensor_slices(data_fids)

    # Convert filenames to actual image tensors.
    dataset = dataset.map(
        lambda fid: common.fid_to_image(
            fid, tf.constant('dummy'), image_root=args.image_root,
            image_size=pre_crop_size if args.crop_augment else net_input_size),
        num_parallel_calls=args.loading_threads)

    # Augment the data if specified by the arguments.
    # `modifiers` is a list of strings that keeps track of which augmentations
    # have been applied, so that a human can understand it later on.
    modifiers = ['original']
    if args.flip_augment:
        dataset = dataset.map(flip_augment)
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

    if args.crop_augment == 'center':
        dataset = dataset.map(lambda im, fid, pid:
            (five_crops(im, net_input_size)[0], fid, pid))
        modifiers = [o + '_center' for o in modifiers]
    elif args.crop_augment == 'five':
        dataset = dataset.map(lambda im, fid, pid: (
            tf.stack(five_crops(im, net_input_size)),
            tf.stack([fid]*5),
            tf.stack([pid]*5)))
        dataset = dataset.apply(tf.contrib.data.unbatch())
        modifiers = [o + m for o in modifiers for m in [
            '_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
    elif args.crop_augment == 'avgpool':
        modifiers = [o + '_avgpool' for o in modifiers]
    else:
        modifiers = [o + '_resize' for o in modifiers]

    emb_model = EmbeddingModel(args)

    # Group it back into PK batches.
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.map(lambda im, fid, pid: (emb_model.preprocess_input(im), fid, pid))
    # Overlap producing and consuming.
    dataset = dataset.prefetch(1)
    tf.keras.backend.set_learning_phase(0)


    with h5py.File(args.filename, 'w') as f_out:

        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=emb_model)
        manager = tf.train.CheckpointManager(ckpt, osp.join(args.experiment_root, 'tf_ckpts'),max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        emb_storage = np.zeros(
            (len(data_fids) * len(modifiers), args.embedding_dim), np.float32)

        # for batch_idx,batch in enumerate(dataset):
        dataset_iter = iter(dataset)
        for start_idx in count(step=args.batch_size):

            try:
                images, _, _ = next(dataset_iter)
                emb = emb_model(images)
                emb_storage[start_idx:start_idx + len(emb)] += emb
                print('\rEmbedded batch {}-{}/{}'.format(
                    start_idx, start_idx + len(emb), len(emb_storage)),
                    flush=True, end='')
            except StopIteration:
                break  # This just indicates the end of the dataset.


        if not args.quiet:
            print("Done with embedding, aggregating augmentations...", flush=True)

        if len(modifiers) > 1:
            # Pull out the augmentations into a separate first dimension.
            emb_storage = emb_storage.reshape(len(data_fids), len(modifiers), -1)
            emb_storage = emb_storage.transpose((1,0,2))  # (Aug,FID,128D)

            # Store the embedding of all individual variants too.
            emb_dataset = f_out.create_dataset('emb_aug', data=emb_storage)

            # Aggregate according to the specified parameter.
            emb_storage = AGGREGATORS[args.aggregator](emb_storage)

        # Store the final embeddings.
        emb_dataset = f_out.create_dataset('emb', data=emb_storage)


        # Store information about the produced augmentation and in case no crop
        # augmentation was used, if the images are resized or avg pooled.
        f_out.create_dataset('augmentation_types', data=np.asarray(modifiers, dtype='|S'))

if __name__ == '__main__':



    arg_experiment_root = const.experiment_root_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    for subset in ['test']:
        exp_dir = 'cub_densenet_direct_normalize_npairs_loss_m_0.2'
        folder_name = 'emb'
        dataset_name = 'cub'
        if dataset_name == 'cub':
            csv_file = 'cub'
        elif dataset_name == 'inshop':
            csv_file = 'deep_fashion'
        elif dataset_name == 'stanford':
            csv_file = 'stanford_online'
        else:
            raise NotImplementedError('dataset {} not valid'.format(dataset_name))

        args = [
            '--experiment_root', arg_experiment_root + exp_dir,
            '--dataset', './data/'+csv_file+'_'+subset+'.csv',
            '--filename', subset+'_embeddings_augmented.h5',
            '--foldername',folder_name,
            '--crop_augment','center', ## Make sure it follows the training resolution
            # '--batch_size','40',
        ]
        main(args)

