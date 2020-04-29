import logging
import os
import argparse
import codecs
import time
import numpy as np
import tqdm
import pandas
from scipy.sparse import coo_matrix

from implicit.als import AlternatingLeastSquares
from implicit.approximate_als import (AnnoyAlternatingLeastSquares, FaissAlternatingLeastSquares,
                                      NMSLibAlternatingLeastSquares)
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,
                                         TFIDFRecommender, bm25_weight)
from implicit.evaluation import ranking_metrics_at_k, train_test_split

# maps command line model argument to class name
MODELS = {"als":  AlternatingLeastSquares,
          "nmslib_als": NMSLibAlternatingLeastSquares,
          "annoy_als": AnnoyAlternatingLeastSquares,
          "faiss_als": FaissAlternatingLeastSquares,
          "tfidf": TFIDFRecommender,
          "cosine": CosineRecommender,
          "bpr": BayesianPersonalizedRanking,
          "lmf": LogisticMatrixFactorization,
          "bm25": BM25Recommender}


def main():
    # setup args
    parser = argparse.ArgumentParser(description="""Generates recommendations for each user.
Expects the input data to be in format 'user_id\tvideo_id\tweight'.
If only --input provided it is used both for model fit and recommendations;
use --rec_input to provide a separate dataset for recommendations
""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rec_count", type=int, default=300,
                        dest="rec_count", help="num of recommendations to generate")
    parser.add_argument("--limit", type=int, default=0,
                        dest="limit", help="limit num of users to generate recommendations for")
    parser.add_argument("--input", type=str, default="dataset.tsv",
                        dest="inputfile", help="input file name")
    parser.add_argument("--rec_input", type=str, default="",
                        dest="rec_inputfile", help="separate input file name for recommendations")
    parser.add_argument("--output", type=str, default="results.csv",
                        dest="outputfile", help="output file name")
    parser.add_argument("--gpu", action="store_true", help="use GPU (CUDA)")
    parser.add_argument("--evaluate", action="store_true",
                        help="evaluate (cross-validate) model after training")
    parser.add_argument("--model", type=str, default="als",
                        dest="model", help="model to calculate (%s)" % "/".join(MODELS.keys()))
    parser.add_argument("--log", type=str, default="DEBUG",
                        dest="log", help="logging level: CRITICAL|ERROR|WARNING|INFO|DEBUG")
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=args.log)

    # setup model
    model = get_model(args.model, args.gpu)

    # read dataset
    users, videos, weight_coo = read_dataset(args.inputfile)

    # prepate
    model, weight_coo = prepate(model, weight_coo)

    # if evaluation flag is set
    if args.evaluate:
        # need to split the dataset into (train, test) parts for cross-validation
        train_coo, test_coo = train_test_split(
            weight_coo, train_percentage=0.8)
        # convert test matrix to CSR
        test_csr = test_coo.tocsr()
    else:
        # otherwise use the whole `inputfile` dataset for training
        train_coo = weight_coo

    # convert train matrix to CSR
    train_csr = train_coo.tocsr()

    # train
    trained_model = train(model, train_csr)

    # evaluate
    if args.evaluate:
        evaluate(trained_model, train_csr, test_csr)

    # read the dataset for recommendations if it's different from the training one
    if args.rec_inputfile is not "" and args.rec_inputfile is not args.inputfile:
        users, videos, weight_coo = read_dataset(args.rec_inputfile)
        # prepare matrix
        model, weight_coo = prepate(model, weight_coo)

    # recommend
    calculate_recommendations(trained_model,
                              users, videos, weight_coo.tocsr(), args.outputfile, args.rec_count, limit=args.limit)

    logging.info("Recommendations are written to '%s'", args.outputfile)


def get_model(model_name, use_gpu=False):
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)

    # some default params as suggested by the author of Implicit
    if issubclass(model_class, AlternatingLeastSquares):
        params = {"factors": 16, "dtype": np.float32, "use_gpu": use_gpu}
    elif model_name == "bm25":
        params = {"K1": 100, "B": 0.5}
    elif model_name == "bpr":
        params = {"factors": 63, "use_gpu": use_gpu}
    elif model_name == "lmf":
        params = {"factors": 30, "iterations": 40, "regularization": 1.5}
    else:
        params = {}

    return model_class(**params)


def read_dataset(filename):
    """ Reads the original dataset TSV as a pandas dataframe """

    # read in triples of user_id/video_id/weight from the input dataset
    logging.info("Reading dataset '%s'...", filename)
    start = time.time()
    data = pandas.read_table(filename,
                             usecols=[0, 1, 2],
                             names=["user", "video", "weight"],
                             na_filter=False)

    # map each video and user to a unique numeric value
    data["user"] = data["user"].astype("category")
    data["video"] = data["video"].astype("category")

    # create a sparse CSR matrix
    weight = coo_matrix((data["weight"].astype(np.float32),
                         (data["video"].cat.codes.copy(),
                          data["user"].cat.codes.copy())))

    logging.debug("Read data file in %0.2fs", time.time() - start)

    return np.array(data["user"].cat.categories), np.array(data["video"].cat.categories.astype(np.str)), weight


def train(model, weight_csr):
    """ Trains the model """

    logging.debug("Training model %s", model.__class__)
    start = time.time()
    model.fit(weight_csr)
    logging.debug("Trained model in %0.2fs", time.time() - start)

    return model


def evaluate(trained_model, train_csr, test_csr):
    """ Evaluates the model """

    logging.debug("Evaluating model...")
    start = time.time()

    m = ranking_metrics_at_k(trained_model, train_csr.T.tocsr(),
                             test_csr.T.tocsr(), K=100, num_threads=0)
    logging.debug("Evaluated in in %0.2fs", time.time() - start)
    logging.info("Evaluation metrics: %s", m)


def calculate_recommendations(model, users, videos, weight_csr, output_filename, recs_count=10, limit=0):
    """ Generates video recommendations for each user in the dataset """

    max_users = limit if limit > 0 else len(users)

    logging.info("Building recommendations for %s users with model of '%s'...",
                 max_users, model.__class__)
    start = time.time()
    weight = weight_csr.T.tocsr()
    with tqdm.tqdm(total=max_users) as progress:
        with codecs.open(output_filename, "w", "utf8") as o:
            c = 0
            for user_idx, user_id in enumerate(users):
                # if limit is set, respect it
                if c >= max_users:
                    break
                c += 1
                video_ids = ""
                for video_idx, _score in model.recommend(user_idx, weight, N=recs_count):
                    video_ids += " " + videos[video_idx]
                o.write("%s%s\n" %
                        (user_id, video_ids))
                progress.update(1)
    logging.debug("Generated recommendations for %s users in %0.2fs",
                  max_users, time.time() - start)


def prepate(model, weight_coo):
    """ Prepares a model and a weight matrix in case of ALS """

    # if we're training an ALS based model, weight input by bm25
    if issubclass(model.__class__, AlternatingLeastSquares):
        # lets weight these models by bm25weight.
        logging.debug("Weighting matrix by bm25_weight")
        weight_coo = bm25_weight(weight_coo, K1=100, B=0.8)
        # also disable building approximate recommend index
        model.approximate_similar_items = False
    return model, weight_coo


if __name__ == "__main__":
    main()
