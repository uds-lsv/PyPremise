"""
Copyright (c) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Mapping
import csv

from .core import PremiseInstance, Direction


def from_sparse_index_lists(indices: List[List[int]], labels: List[int]) -> List[PremiseInstance]:
    """
    Creates PremiseInstances from list of index-lists and corresponding misclassifications labels
    Each index-list contains the items/sparse features of one instance/transaction.

    The labels list contains the (mis)classification labels, one for each instance. 0 means misclassification
    and 1 means correct classification.

    Example: for
       an instance with features 0,3 that was correctly classified (label=1) and
       an instance with features 1,3,6 that was misclassified (label=0)
    the parameters would look like

    indices = [ [0, 3], [1, 3, 6]]
    labels = [0, 1]
    """
    if len(indices) != len(labels):
        raise Exception(f"Index lists and labels need to have the same length, not {len(indices)} and {len(labels)}.")

    instances = []
    for index_list, label in zip(indices, labels):
        for index in index_list:
            if not isinstance(index, int):
                raise Exception(f"Index must be an int, not {index}.")
        index_list = sorted(index_list)

        if label == 0:
            label = Direction.MISCLASSIFICATION
        elif label == 1:
            label = Direction.CORRECT
        else:
            raise Exception(f"Label must be 0 (= misclassification) or 1 (correct classification), not {label}.")
        instances.append(PremiseInstance(index_list, label))
    return instances


def from_dense_index_matrix(indices, labels) -> List[PremiseInstance]:
    """
    Creates PremiseInstances from a dense index matrix and the corresponding misclassifications labels
    In the matrix, each row is one one instance/transaction. Each column represents a feature. If
    a feature is present in an instance, the corresponding entry is 1, else it is 0.

    The labels contains the (mis)classification labels, one for each instance. 0 means misclassification
    and 1 means correct classification.

    Example: for
       an instance with features 0,3 that was correctly classified (label=1) and
       an instance with features 1,2,4 that was misclassified (label=0)
    the parameters would look like

    indices = [ [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1] ]
    labels = [0, 1]
    """
    sparse_indices = []
    for row in indices:
        sparse_row = []
        for column, value in enumerate(row):
            if value == 0:
                continue
            elif value == 1:
                sparse_row.append(column)
            else:
                raise Exception(f"A binary dense feature matrix must only contain 0 and 1 as values, not {value}.")
        sparse_indices.append(sparse_row)
    return from_sparse_index_lists(sparse_indices, labels)


def from_token_lists(token_lists: List[List[str]], labels: List[int]) -> (List[PremiseInstance],
                                                                          Mapping[str, int], Mapping[int, str]):
    """
    Creates PremiseInstances from list of token-lists and corresponding misclassifications labels
    Each token-list contains the tokens of one instance.

    The labels list contains the (mis)classification labels, one for each instance. 0 means misclassification
    and 1 means correct classification.

    Example: for
       an instance with tokens "a", "brown", "dog" that was correctly classified (label=1) and
       an instance with features "a", "black", "cat" that was misclassified (label=0)
    the parameters would look like

    token_lists = [ ["a", "brown", "dog"], ["a", "black", "cat"]]
    labels = [0, 1]

    The tokens get converted into index lists (on which Premise works). Vocabulary mappings are available to convert
    from and to this index representation.

    Returns the list of PremiseInstance objects, a vocabulary to map the tokens to their index representations and
    a reversed vocabulary to map an index to its token.
    """
    voc_token_to_index = {}
    index_lists = []
    for tokens in token_lists:
        indices = []
        for token in tokens:
            if not token in voc_token_to_index:
                voc_token_to_index[token] = len(voc_token_to_index)
            index = voc_token_to_index[token]
            indices.append(index)
        indices = set(indices)  # remove duplicate indices (words appearing twice in a sentence)
        indices = sorted(indices)  # bag of word assumption, so we can order the indices, as expected by Premise input
        index_lists.append(indices)

    instances = from_sparse_index_lists(index_lists, labels)
    voc_index_to_token = {v:k for k,v in voc_token_to_index.items()}

    return instances, voc_token_to_index, voc_index_to_token


def from_csv_sparse_index_file(path_features: str, path_labels: str, delimiter=" ") -> List[PremiseInstance]:
    """
    Creates PremiseInstances from one file containing the features (in a sparse format) and one file
    containing the labels. Each row represents one instance/transaction.

    The features are stored in a sparse format, i.e. only those features that occur in an instance are mentioned.
    For the labels, 0 means misclassification and 1 means correct classification.

    Example: for
      an instance with features 0,3 that was correctly classified (label=1) and
      an instance with features 1,3,6 that was misclassified (label=0)
    the files would look like (with separator " ")

    features.dat:
    0 3
    1 3 6

    labels.dat:
    1
    0

    :param path_features: path to the features file
    :param path_labels: path to the labels file
    :param delimiter: separator of the features
    :return: a list of PremiseInstance objects, one for each row
    """
    index_lists = _read_indices_from_csv(path_features, delimiter)
    labels = _read_labels_from_file(path_labels)
    return from_sparse_index_lists(index_lists, labels)


def from_csv_dense_index_file(path_features: str, path_labels: str, delimiter=" ") -> List[PremiseInstance]:
    """
    Creates PremiseInstances from a file containing the features (in a dense format) and one file containing the labels.
    Each row represents one instance/transaction.

    The features are stored in a dense format, i.e. for each feature in a row, if the feature is present,
    a 1 is given. And if it is not present, a 0 is given. All rows need to have the same number of entries/features.
    For the labels, 0 means misclassification and 1 means correct classification.

    Example: for
      an instance with features 0,3 that was correctly classified (label=1) and
      an instance with features 1,3,6 that was misclassified (label=0)
    the files would look like (with separator " ")

    features.dat:
    1 0 0 1 0 0 0
    0 1 0 1 0 0 1

    labels.dat:
    1
    0

    :param path_features: path to the features file
    :param path_labels: path to the labels file
    :param delimiter: separator of the features
    :return: a list of PremiseInstance objects, one for each row
    """
    index_lists = _read_indices_from_csv(path_features, delimiter)
    labels = _read_labels_from_file(path_labels)
    return from_dense_index_matrix(index_lists, labels)


def from_tokenized_file(path_features: str, path_labels: str, delimiter=" ") -> (List[PremiseInstance],
                                                                                 Mapping[str, int], Mapping[int, str]):
    """
    Creates PremiseInstances from a file containing the tokenized words (features) and one file containing the
    misclassification labels. Each row represents one instance.

    Example: for
       an instance with tokens "a", "brown", "dog" that was correctly classified (label=1) and
       an instance with features "a", "black", "cat" that was misclassified (label=0)
    the parameters would look like (with tokenization delimiter " ")

    features.dat:
     a brown dog
     a black cat

    labels.dat:
     1
     0

     :param path_features: path to the feature tokens file
     :param path_labels: path to the labels file
     :param delimiter: separator of the tokens
     :return: a list of PremiseInstance objects, one for each row, and a Mapping from tokens to indices (used by
                Premise) and a reverse Mapping from indices to tokens.
    """
    tokens = _read_features_from_csv(path_features, delimiter)
    labels = _read_labels_from_file(path_labels)
    return from_token_lists(tokens, labels)


def _read_indices_from_csv(path_features: str, delimiter: str):
    """
    Helper method to load indices from csv files.
    """
    index_lists = []
    all_features = _read_features_from_csv(path_features, delimiter)
    for row_nbr, row in enumerate(all_features):
        indices = []
        for c in row:
            try:
                index = int(c)
            except:
                raise Exception(f"Could not read the features file because in line {row_nbr} the value was not a "
                                f"number, but {c}")

            indices.append(index)
        index_lists.append(indices)
    return index_lists


def _read_features_from_csv(path_features, delimiter):
    """
    Helper method to read feature files. The specific from_csv_*
    then implements how to treat the features (e.g. convert to int indices)
    """
    with open(path_features, "r") as in_file:
        reader = csv.reader(in_file, delimiter=delimiter)
        features = [line for line in reader]
        return features


def _read_labels_from_file(path_labels):
    """
    Helper method to read the labels file
    """
    labels = []
    with open(path_labels, "r") as in_file:
        for i, line in enumerate(in_file):
            line = line.strip()
            try:
                label = int(line)
            except:
                raise Exception(f"Could not read the labels file because in line {i} the value was not a "
                                f"number, but {line}.")
            labels.append(label)
    return labels


def create_fasttext_mapping(path_fasttext: str, voc_index_to_token: Mapping[int, str], verbose=True):
    """

    :param path_fasttext: path to the downloaded fasttext model (e.g. cc.en.300.bin). Needs to be the binary FastText
     model!
    :param voc_index_to_token: A map mapping the indices (used by Premise) to their corresponding tokens.
    :return: a mapping between indices and vector embedding as well as the dimensionality of the embedding vectors
    """
    try:
        import fasttext
    except Exception:
        raise Exception("Could not load Python module fasttext. This module is optional and "
                        "only needed for create_fasttext_mapping. Did you install it? If not, please follow the"
                        "instructions in the README.")
    if verbose:
        print("Loading FastText model, this might take a bit.")
    embedding = fasttext.load_model(path_fasttext)
    dimensionality = embedding.get_dimension()

    if verbose:
        print("FastText loaded. Mapping the tokens to their embeddings.")
    max_index = max(voc_index_to_token.keys())
    embedding_index_to_vector = {}
    for i in range(max_index+1):
        if i not in voc_index_to_token:
            raise Exception(f"Index {i} is not in voc_index_to_token. The mapping needs to contain all indices"
                            f"between 0 and {max_index} (the maximum index in voc_index_to_token) even if"
                            f"an index never appears in the data.")
        embedding_index_to_vector[i] = embedding[voc_index_to_token[i]]
    return embedding_index_to_vector, dimensionality


def get_dummy_data():
    """
    Some dummy example data that should find "How many" as a misclassification pattern,
    "When was taken" as correct classification pattern and no pattern about "ducks" as
    these are not informative about the classification status.

    :return:
    """
    sentences = ["How many ducks are there",
                 "How many roosters are in the puddle",
                 "How many ducks do you see",
                 "How many ducks are there",
                 "How many chickens are crossing the road",
                 "What are the ducks eating",
                 "How many ducks does one need",
                 "How many ducks and chickens are there",
                 "Are there any ducks",
                 "Where is the rooster looking at?",
                 "How many chickens are there",
                 "How many roosters can you see",

                 "When was the photo taken",
                 "Are there many ducks playing",
                 "When was the photograph taken",
                 "When was this photo taken",
                 "When did the photographer take this photograph",
                 "Do you see any ducks",
                 "Can you see a rooster in the picture",
                 "Can you see ducks in the photograph",
                 "When do you think was the photo taken",
                 "When was the photo with the ducks taken",
                 "When was the photograph taken",
                 "When was the photograph taken where one can see the rooster"]
    sentences = [s.split(" ") for s in sentences]
    labels = [0] * 12 + [1] * 12
    return from_token_lists(sentences, labels)
