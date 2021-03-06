import multiprocessing
import re
import os
import subprocess
import pickle
from functools import partial
import cv2
import numpy as np
from scipy import signal as ss

from benchmarks1 import (
    call_benchmark,
    default_params,
    tuned_network_path,
)
from deprecated_measure_segmentation import (
    score_splitter,
    tsai_bars,
)
# import measure_segmentation

from score_retrieval.constants import (
    arguments,
    NONE_ALG,
    DATA_DIR,
    DEFAULT_DATASET,
)
from score_retrieval.data import (
    index_images,
    gen_label_name_index,
    get_label,
    load_img,
    database_paths,
    query_paths,
)


def resample_arr(arr, resample_len):
    """Resample array to constant length."""
    out_arr = ss.resample(np.asarray(arr), resample_len)
    assert out_arr.shape == (resample_len,), "{}.shape == {} != ({},)".format(
        out_arr, out_arr.shape, resample_len)
    return out_arr


def normalize_arr(arr):
    """Normalize array to constant mean and stdev."""
    return (arr - np.mean(arr))/np.std(arr)


def isnull(arr):
    """Determine whether the given array is null."""
    return not arr.shape or sum(arr.shape) == 0


def get_veclist_path(img_path, alg):
    """Get the veclist path for the given image path and alg."""
    if alg == NONE_ALG:
        alg = None
    base_path = os.path.splitext(img_path)[0]
    if alg is None:
        return "{}.npy".format(base_path)
    else:
        return "{}_{}.npy".format(base_path, alg)


def parallelize_veclist(startIdx, endIdx):
    startIdx = int(startIdx)
    endIdx = int(endIdx)
    fileList = "/home/dhyang/Evan_CNNRetrieval/score-retrieval1/score_retrieval/fileList_ref.txt"
    basepath = "/pylon5/ir5phqp/dhyang/ref_jpgs/"
    fnames = []
    with open(fileList, 'r') as f:
        for fname in f:
            fnames.append(fname.strip())
    if endIdx < len(fnames):
        fnames = fnames[startIdx:endIdx]
    else:
        fnames = fnames[startIdx:]
    fnames1 = fnames
    for fname in fnames:
        vec_path = os.path.splitext(fname)[0]+"_benchmark.npy"
        vec_path = vec_path[len(basepath):]
        vec_path = os.path.join(
            "/pylon5/ir5phqp/dhyang/ref_RMAC/", vec_path)
        print(vec_path)
        if os.path.exists(vec_path):
            print("EXISTS", fname)
            fnames1.remove(fname)
    print(fnames1)
    for fn in fnames1:
        extractVector(fn)


def extractVector(path):
    normalize = False
    resample_len = None
    debug = False
    grayscale = False
    alg_name = 'benchmark'
    spath = "/pylon5/ir5phqp/dhyang/ref_RMAC/"
    veclist_path = get_veclist_path(path, alg_name)
    print("Generating veclist for {}...".format(veclist_path))
    DATA_DIR = "/pylon5/ir5phqp/dhyang/ref_jpgs/"
    DEFAULT_DATASET = ""
    new_path = spath + \
        os.path.relpath(veclist_path, os.path.join(
            DATA_DIR, DEFAULT_DATASET))
    print(new_path)
    if os.path.exists(new_path):
        print("---------------------------", new_path, "Already Exists")
        return
    image = load_img(path, grayscale=grayscale)
    if image is None:
        print("Got None for imread({}).".format(path))
        return
    if debug:
        print("image.shape =", image.shape)
    raw_veclist = make_benchmark_vec(image)
    if raw_veclist is None:
        print("Got None raw_veclist from image {}.".format(path))
        return
    veclist = []
    for vec in raw_veclist:
        if not isnull(vec):
            if resample_len is not None:
                vec = resample_arr(vec, resample_len)
            if normalize:
                vec = normalize_arr(vec)
            veclist.append(vec)
    veclist = np.asarray(veclist)

    if isnull(veclist):
        print("Got null veclist for {} with shape {} (raw len {}).".format(
            path, veclist.shape, len(raw_veclist)))
        return
    if debug:
        print("veclist.shape =", veclist.shape)
    if not os.path.exists(os.path.dirname(new_path)):
        os.makedirs(os.path.dirname(new_path))
    print("Saved to", new_path)
    np.save(new_path, veclist)


def save_veclists(image_paths, image_to_veclist_func, alg_name, grayscale=True, resample_len=None, normalize=False, debug=False):
    """Saves database of vectors using the given vector generation function."""
    spath = "/pylon5/ir5phqp/dhyang/CNN_Features"

    for path in image_paths:
        print(path)
        subprocess.call(['sbatch', 'bootlegWrapper.sh', path])


def load_veclist(image_path, alg_name):
    """Return veclist or None for the given image path."""
    veclist_path = get_veclist_path(image_path, alg_name)
    if os.path.exists(veclist_path):
        print("Loading {}...".format(veclist_path))
        return np.load(veclist_path)
    else:
        print("Skipping {}...".format(veclist_path))
        return None


def load_query_veclists(query_paths, alg_name):
    """Return q_labels, q_veclists."""
    q_labels = []
    q_veclists = []
    for path in query_paths:
        veclist = load_veclist(path, alg_name)
        if veclist is not None:
            label = get_label(path, True)
            q_labels.append(label)
            q_veclists.append(veclist)
    return q_labels, q_veclists


def load_db_vecs(database_paths, alg_name, return_paths=False, ind_per_page=False):
    """Return db_labels, db_vecs, db_inds."""
    db_labels = []
    db_vecs = []
    db_indices = []
    if return_paths:
        db_paths = []

    # generate db index
    db_index = []
    for path in database_paths:
        db_index.append((get_label(path), path))

    # sort images into groups based on their order in their piece
    base_index = gen_label_name_index(db_index)
    for label, name_index in base_index.items():
        for name, paths in name_index.items():
            # for each sequence of sequential images put the vectors
            #  in the database with the right index
            i = 0
            for img_path in paths:
                if ind_per_page:
                    i = 0
                veclist = load_veclist(img_path, alg_name)
                if veclist is None:
                    i += 1
                else:
                    for vec in veclist:
                        db_labels.append(label)
                        db_vecs.append(vec)
                        db_indices.append(i)
                        if return_paths:
                            db_paths.append(img_path)
                        i += 1

    if return_paths:
        return db_labels, db_vecs, db_indices, db_paths
    else:
        return db_labels, db_vecs, db_indices


def make_benchmark_vec(image):
    """Make vector using the benchmark algorithm."""
    resized_image = cv2.resize(image, (1024, 1024))
    print("Shape:", resized_image.shape)
    return call_benchmark(images=[resized_image])


def func_with_cnn_params(func, **params):
    """Return function that sets cnn params first."""
    def new_func(*args, **kwargs):
        old_default_params = default_params.copy()
        default_params.update(params)
        try:
            return func(*args, **kwargs)
        finally:
            default_params.update(old_default_params)
    return new_func


ALGS = {
    "benchmark": (
        make_benchmark_vec,
        dict(grayscale=False),
    ),
}


def generate_vectors_from_args(parsed_args=None):
    """Save veclists for the given alg and args."""
    if parsed_args is None:
        parsed_args = arguments.parse_args()
    print("ALG:", parsed_args.alg)

    func, kwargs = ALGS[parsed_args.alg]
    if parsed_args.multidataset:
        paths = database_paths + query_paths
    else:
        paths = [path for label, path in index_images(parsed_args.dataset)]
    print("FUNC:", func)
    save_veclists(paths, func, parsed_args.alg, **kwargs)
