import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from score_retrieval.constants import arguments
from score_retrieval.data import gen_data_from_args
from score_retrieval.vec_db import generate_vectors_from_args
from score_retrieval.retrieval import run_retrieval_from_args


def main():
    """Do vector generation and retrieval."""
    parsed_args = arguments.parse_args()

    ## generate vectors
    generate_vectors_from_args(parsed_args)



if __name__ == "__main__":
    main()
