import argparse
import os
import time
import math
import pickle
from pprint import pprint

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print, compute_mrr, compute_acc
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = ['oxford5k,paris6k', 'roxford5k,rparis6k', 'oxford5k,paris6k,roxford5k,rparis6k', 'scores']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k', 'scores']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help='network path, destination where network is saved')
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help='network off-the-shelf, in the format ARCHITECTURE-POOLING or ARCHITECTURE-POOLING-whiten,' +
                    ' examples: resnet101-gem | resnet101-gem-whiten')

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                   help='comma separated list of test datasets: ' +
                        ' | '.join(datasets_names) +
                        ' (default: oxford5k,paris6k)')
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help='maximum size of longer image side used for testing (default: 1024)')
parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true',
                    help='use multiscale vectors for testing')
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help='dataset used to learn whitening for testing: ' +
                        ' | '.join(whitening_names) +
                        ' (default: None)')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for testing (default: 0)')


def load_network(network_path):
    """Load network from a path."""
    print(">> Loading network:\n>>>> '{}'".format(network_path))
    print("wldkfjwifjlwidofj")
    if network_path in PRETRAINED:
        # pretrained networks (downloaded automatically)
        state = load_url(PRETRAINED[network_path], model_dir=os.path.join(get_data_root(), 'networks'))
    else:
        state = torch.load(network_path)
    net = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'],
                        mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
    net.load_state_dict(state['state_dict'])

    # if whitening is precomputed
    if 'Lw' in state['meta']:
        net.meta['Lw'] = state['meta']['Lw']

    print(">>>> loaded network: ")
    print(net.meta_repr())

    return net


def load_offtheshelf(network_name):
    """Load off the shelf network."""
    offtheshelf = network_name.split('-')
    if len(offtheshelf)==3:
        if offtheshelf[2]=='whiten':
            offtheshelf_whiten = True
        else:
            raise(RuntimeError("Incorrect format of the off-the-shelf network. Examples: resnet101-gem | resnet101-gem-whiten"))
    else:
        offtheshelf_whiten = False

    print(">> Loading off-the-shelf network:\n>>>> '{}'".format(network_name))
    net = init_network(model=offtheshelf[0], pooling=offtheshelf[1], whitening=offtheshelf_whiten)

    print(">>>> loaded network: ")
    print(net.meta_repr())

    return net


def main():
    args = parser.parse_args()

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:
        net = load_network(args.network_path)

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        net = load_offtheshelf(args.network_offtheshelf)

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:

            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))

            if args.multiscale:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:

            print('>> {}: Learning whitening...'.format(args.whitening))

            if args.whitening == "scores":
                # special logic for scores database
                from score_retrieval.exports import (
                    db,
                    train_images as images,
                )

            else:
                # loading db
                db_root = os.path.join(get_data_root(), 'train', args.test_whiten)
                ims_root = os.path.join(db_root, 'ims')
                db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.test_whiten))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

            # extract whitening vectors
            print('>> {}: Extracting...'.format(args.whitening))
            wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)

            # learning whitening
            print('>> {}: Learning...'.format(args.whitening))
            wvecs = wvecs.numpy()
            m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
            Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        if dataset == "scores":
            # Special added logic to handle loading our score dataset
            from score_retrieval.exports import (
                images,
                qimages,
                gnd,
            )

            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, args.image_size, transform, ms=ms, msp=msp)

        else:
            # extract ground truth
            cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
            gnd = cfg['gnd']

            # prepare config structure for the test dataset
            images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs = [tuple(gnd[i]['bbx']) for i in range(cfg['nq'])]

            # extract database and query vectors
            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)

        # validation
        print(">> {}: gnd stats: {}, {}, {}".format(
            dataset,
            len(gnd),
            [len(x["ok"]) for x in gnd[10:]],
            [len(x["junk"]) for x in gnd[10:]],
        ))
        print(">> {}: image stats: {}, {}".format(dataset, len(images), len(qimages)))
        assert len(gnd) == len(qimages), (len(gnd), len(qimages))

        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()
        print(">> {}: qvecs.shape: {}".format(dataset, qvecs.shape))

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        print(">> {}: ranks (shape {}) head: {}".format(dataset, ranks.shape, ranks[10:,10:]))
        print(">> {}: gnd head: {}".format(dataset, gnd[5:]))

        # Compute and print metrics
        compute_acc(ranks, gnd, dataset)
        compute_mrr(ranks, gnd, dataset)
        compute_map_and_print(dataset, ranks, gnd)

        if Lw is not None:
            # whiten the vectors
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_acc(ranks, gnd, dataset + " + whiten")
            compute_mrr(ranks, gnd, dataset + " + whiten")
            compute_map_and_print(dataset + " + whiten", ranks, gnd)

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
