from __future__ import division

import os

import torch
import numpy as np
from torch.utils.model_zoo import load_url
from torchvision import transforms
from torch.autograd import Variable

from cirtorch.utils.general import get_data_root
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.networks.imageretrievalnet import (
    init_network,
    extract_vectors,
    extract_ss,
    extract_ms,
)
from cirtorch.datasets.genericdataset import ImagesFromDataList
from cirtorch.examples.test import (
    load_network,
    load_offtheshelf,
)

from score_retrieval.data import indices_with_label
from score_retrieval.exports import (
    db,
    train_images,
)


# can be imported and used as a network argument
tuned_network_path = os.path.join(
    os.path.dirname(__file__),
    "weights",
    "scores_vgg16_gem_whiten_contrastive_m0.85_adam_lr1.0e-06_wd1.0e-04_nnum2_qsize250_psize2500_bsize1_imsize1024",
    "model_epoch100.pth.tar",
)


def vectors_from_images(net, images, transform, ms=[1], msp=1, print_freq=10, setup_network=True, gpu=False):
    """Extract vectors from images given as a pytorch array."""
    # moving network to gpu and eval mode
    print("gpu:",gpu)
    if setup_network:
        if gpu:
            net.cuda()
        net.eval()
    print("Set up GPU")
    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromDataList(images=images, transform=transform),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    print("Begin Vector Extraction")
    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    print("hello0")
    print(enumerate(loader))
    for i, input_data in enumerate(loader):
        print(gpu)
        if gpu:
            print("hello1")
            input_data = input_data.cuda()
            print("hello2")
        input_var = Variable(input_data)

        # fix tensor shape
        if len(input_var.shape) > 4 and input_var.shape[0] == 1:
            input_var = input_var[0]

        if len(ms) == 1:
            vecs[:, i] = extract_ss(net, input_var)
        else:
            vecs[:, i] = extract_ms(net, input_var, ms, msp)

        if (i+1) % print_freq == 0 or (i+1) == len(images):
            print('{}/{}...'.format((i+1), len(images)), end='')
    print('done')
    return vecs


LEARNED_WHITENING = {}


def get_scores_whitening(whiten_key, net, transform, ms, msp, image_size, setup_network=True, gpu=True):
    """Learn scores whitening for the given network."""
    if whiten_key in LEARNED_WHITENING:
        return LEARNED_WHITENING[whiten_key]

    else:
        print("Learning scores whitening...")

        # extract whitening vectors
        wvecs = extract_vectors(net, train_images, image_size, transform, ms=ms, msp=msp, setup_network=setup_network, gpu=gpu)

        # learning whitening
        wvecs = wvecs.numpy()
        m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
        Lw = {'m': m, 'P': P}

        # cache learned whitening
        LEARNED_WHITENING[whiten_key] = Lw

        print("Whitening learned and cached.")
        return Lw


LOADED_NETWORKS = {}


# We store these in a dict so they can be modified externally.
default_params = dict(
    network="vgg16-spoc",
    offtheshelf=True,
    image_size=1024,
    gpu=True,
    multiscale=True,
    whitening="scores",
    #whitening=None,
)


def call_benchmark(
    # must pass one of images or paths
    images=None,
    paths=None,
    **kwargs,
):
    """Run the given network on the given data and return vectors for it."""
    # load params
    print("sdlkfnwlkfjlwkljwlk")
    params = default_params.copy()
    params.update(kwargs)
    print(params)

    network = params["network"]
    offtheshelf = params["offtheshelf"]
    image_size = params["image_size"]
    gpu = params["gpu"]
    multiscale = params["multiscale"]
    whitening = params["whitening"]
    net_key = (network, offtheshelf, gpu)
    print(network,offtheshelf)

    net = init_network(model='vgg16',pooling='gem',whitening=False,pretrained=True)
    if gpu:
        net.cuda()
    net.eval()
    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if multiscale:
        ms = [1, 1/np.sqrt(2), 1/2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std'],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # process the given data
    if images is not None:
        images = np.asarray(images)
        print("images.shape =", images.shape)
        vecs = vectors_from_images(net, images, transform, ms=ms, msp=msp, setup_network=False, gpu=gpu)
    else:
        vecs = extract_vectors(net, paths, image_size, transform, ms=ms, msp=msp, setup_network=False, gpu=gpu)

    # convert to numpy
    vecs = vecs.numpy()

    # take transpose
    vecs = vecs.T
    print("vecs.shape =", vecs.shape)
    return vecs
