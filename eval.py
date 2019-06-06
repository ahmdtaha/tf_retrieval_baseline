import sys
sys.path.append('..')
sys.path.append('/vulcan/scratch/ahmdtaha/libs/kmcuda/src')
import numpy as np
import common
import os
from argparse import ArgumentParser, FileType
from ranking import METRIC_CHOICES
import h5py
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from libKMCUDA import kmeans_cuda
from scipy.spatial.distance import pdist
import logging.config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
parser = ArgumentParser(description='Evaluate a ReID embedding.')
parser.add_argument(
    '--excluder', required=True, choices=('market1501', 'diagonal','duke'),
    help='Excluder function to mask certain matches. Especially for multi-'
         'camera datasets, one often excludes pictures of the query person from'
         ' the gallery if it is taken from the same camera. The `diagonal`'
         ' excluder should be used if this is *not* required.')


parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset csv file.')

parser.add_argument(
    '--gallery_embeddings', required=True,
    help='Path to the h5 file containing the gallery embeddings.')

parser.add_argument(
    '--metric', required=True, choices=METRIC_CHOICES,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--filename', type=FileType('w'),
    help='Optional name of the json file to store the results in.')

parser.add_argument(
    '--batch_size', default=256, type=common.positive_int,
    help='Batch size used during evaluation, adapt based on your memory usage.')

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    square = np.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * np.dot(x, x.transpose()))
    return np.sqrt(distance_square)


def evaluate_emb(emb, labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(emb)
    #d_mat = pdist(emb)
    # from sklearn.metrics import pairwise_distances
    # x = np.random.rand(3,2)
    # y = np.random.rand(3, 2)
    # print(pairwise_distances(x))
    # print(pairwise_distances(y))
    # print(pairwise_distances(x,y))

    #from eucl_dist import metrics
    #print()
    #d_mat = metrics.euclidean_distances(emb,emb,inverse=False)
    #quit()
    #from eucl_dist.gpu_dist import dist as gdist
    #d_mat = gdist(emb)
    #labels = labels

    names = []
    accs = []
    for k in [1, 2, 4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = np.argpartition(d_mat[i], k)[:k]
            if any(labels[i] == labels[nn] for nn in nns):
                correct += 1
            cnt += 1
        accs.append(correct/cnt)
    return names, accs

def main(argv):
    # Verify that parameters are set correctly.
    args = parser.parse_args(argv)

    gallery_pids, gallery_fids = common.load_dataset(args.gallery_dataset, None)

    log_file = os.path.join(exp_root, "recall_eval")
    logging.config.dictConfig(common.get_logging_dict(log_file))
    log = logging.getLogger('recall_eval')

    with h5py.File(args.gallery_embeddings, 'r') as f_gallery:
        gallery_embs = np.array(f_gallery['emb'])
        #gallery_embs_var = np.array(f_gallery['emb_var'])
        #print('gallery_embs_var.shape =>',gallery_embs_var.shape)

    num_clusters = len(np.unique(gallery_pids))
    print('Start clustering K ={}'.format(num_clusters))
    #kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(gallery_embs)
    #print('NMI :: {}'.format(normalized_mutual_info_score(gallery_pids, kmeans.labels_)))

    centroids, assignments = kmeans_cuda(gallery_embs,num_clusters,seed=3)
    log.info(exp_root)
    log.info('NMI :: {}'.format(normalized_mutual_info_score(gallery_pids, assignments)))
    log.info('Clustering complete')



    log.info('Eval with Recall-K')
    names, accs = evaluate_emb(gallery_embs,gallery_pids)
    log.info(names)
    log.info(accs)

if __name__ == '__main__':

    import getpass
    username = getpass.getuser()
    if username == 'ataha':
        arg_experiment_root = '/mnt/work/datasets/Market-1501-v15.09.15/experiment/'
    elif username == 'ahmedtaha':
        arg_experiment_root = '/Users/ahmedtaha/Documents/dataset/Market-1501-v15.09.15/experiment/'
    elif username == 'ahmdtaha':
        arg_experiment_root = '/vulcan/scratch/ahmdtaha/Market-1501-v15.09.15/experiment/'
    else:
        raise NotImplementedError('Username {} not valid'.format(username))

    dataset_name = 'inshop'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    exp_dir = 'inshop_densenet_fc1024_hard_triplet_m_1.0'
    foldername = 'emb'
    exp_root = os.path.join(arg_experiment_root+exp_dir,foldername)

    if dataset_name == 'cub':
        # db_dir = 'DukeMTMC-reID'
        csv_file = 'cub'
        excluder = 'diagonal'
    elif dataset_name == 'inshop':
        csv_file = 'deep_fashion'
        excluder = 'diagonal'
    else:
        raise  NotImplementedError('dataset {} not valid'.format(dataset_name))


    argv = [
        #'--excluder' ,'duke',
        '--excluder',excluder ,
        '--gallery_dataset','./data/'+csv_file+'_test.csv',
        '--gallery_embeddings',os.path.join(exp_root ,'test_embeddings_augmented.h5'),
        '--metric','euclidean',
        '--filename',os.path.join(exp_root ,'market1501_evaluation.json'),
    ]
    main(argv)


