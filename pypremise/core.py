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

from typing import List, Mapping, Optional, Union
from enum import Enum
import tempfile
import time


class Direction(Enum):
    MISCLASSIFICATION = 0
    CORRECT = 1


class PremiseInstance:

    def __init__(self, features: List[int], label: Direction):
        for i in range(len(features)-1):
            if features[i] >= features[i+1]:
                raise Exception(f"Features need to be ordered, but Item {i} (value = {features[i]}) >= "
                                f"Item {i+1} (value = {features[i+1]}).")
        self.features = features
        self.label = label


class Pattern:
    """
    A pattern is a conjunction of disjunctive items,
    e.g. (1 or 2) and (3) and (4 or 5 or 6)
    """

    def __init__(self, clauses: List[List[Union[int, str]]]):
        self.clauses: List[List[Union[int, str]]] = clauses

    def __str__(self):
        representation = " and ".join(["(" + "-or-".join([str(item) for item in clause]) + ")" for clause in self.clauses])
        return f"{representation}"


class PremiseResult:

    def __init__(self, pattern: Pattern, direction: Direction, label0_count: int, label1_count: int, mdl_gain: float,
                 fisher_value: float):
        self.pattern: Pattern = pattern
        self.direction: Direction = direction
        self.label0_count = label0_count
        self.label1_count = label1_count
        self.mdl_gain: float = mdl_gain
        self.fisher_value: float = fisher_value

    def direction_text(self):
        return "misclassification" if self.direction == Direction.MISCLASSIFICATION else "correct classification"

    def __str__(self):
        return f"{self.pattern} towards {self.direction_text()} (Instances: {self.label0_count} misclassified, " \
               f"{self.label1_count} correctly classified)"


class Premise:

    def __init__(self, voc_index_to_token: Optional[Mapping[int, str]] = None,
                 embedding_index_to_vector: Optional = None, embedding_dimensionality: int = -1,
                 max_neighbor_distance: int = 0,
                 fisher_p_value: float = 0.01, clause_max_overlap: float = 0.05, min_overlap: float = 0.3,
                 verbose: bool = True):

        if embedding_index_to_vector is not None or embedding_dimensionality > 0 or max_neighbor_distance > 0:
            # if we use embeddings, all values need to be set correctly
            if not (embedding_index_to_vector is not None and embedding_dimensionality > 0 and
                    max_neighbor_distance > 0):
                raise Exception("If you use embeddings, you must set all three parts correctly:"
                                "embedding_token_to_vector must be a map from tokens to vectors, the "
                                "embedding dimensionality must be given and the max_neighbor_distance must be > 0.")
        self.voc_index_to_token = voc_index_to_token
        self.embedding_index_to_vector = embedding_index_to_vector
        self.embedding_dimensionality = embedding_dimensionality
        self.max_neighbor_distance = max_neighbor_distance
        self.fisher_p_value = fisher_p_value
        self.clause_max_overlap = clause_max_overlap
        self.min_overlap = min_overlap
        self.verbose = verbose

    def find_patterns(self, instances: List[PremiseInstance]):
        import pypremise.io
        # the Premise C++ code reads and write to files for in and output
        feature_file = tempfile.NamedTemporaryFile()
        label_file = tempfile.NamedTemporaryFile()
        result_file = tempfile.NamedTemporaryFile()
        pypremise.io.write_dat_content(instances, feature_file.name, label_file.name)

        # embeddings
        if self.embedding_index_to_vector is not None:
            embedding_file = tempfile.NamedTemporaryFile()
            embedding_path = embedding_file.name
            max_feature_index = Premise._get_max_feature_index(instances)
            pypremise.io.write_embedding_file(self.embedding_index_to_vector, embedding_path,
                                            self.embedding_dimensionality, max_feature_index)
        else:
            embedding_file = None
            embedding_path = ""

        # actual Premise
        start_time = time.time()
        pypremise.io.call_premise_program(feature_file.name, label_file.name, result_file.name, embedding_path,
                                          self.embedding_dimensionality, self.max_neighbor_distance,
                                          self.fisher_p_value, self.clause_max_overlap, self.min_overlap,
                                          self.verbose)
        if (self.verbose):
            print(f"Premise ran for {time.time() - start_time} seconds.")

        results = pypremise.io.parse_premise_result(result_file.name)

        # clean up temporary files
        feature_file.close()
        label_file.close()
        result_file.close()
        if embedding_file is not None:
            embedding_file.close()

        # if we have a map from indices to tokens, use it to convert our patterns indices to tokens
        if self.voc_index_to_token is not None:
            self._pattern_indices_to_tokens(results)

        return results

    def _pattern_indices_to_tokens(self, results: List[PremiseResult]):
        """
        Converts the features in the given patterns from their index representation to their token representation.
        self.voc_index_to_token needs to be set.
        """
        for result in results:
            pattern = result.pattern
            new_clauses = []
            for clause in pattern.clauses:
                new_clause = []
                for item in clause:
                    if item not in self.voc_index_to_token:
                        raise Exception(f"Index {item} is not in the voc_index_to_token map. Can not convert the"
                                        f"pattern's index representation to tokens. Set voc_index_to_token=None to"
                                        f"not do this conversion.")
                    new_clause.append(self.voc_index_to_token[item])
                new_clauses.append(new_clause)
            pattern.clauses = new_clauses

    @staticmethod
    def _get_max_feature_index(instances: List[PremiseInstance]) -> int:
        max_feature = -1
        for instance in instances:
            for feature in instance.features:
                if feature > max_feature:
                    max_feature = feature
        assert max_feature > -1
        return max_feature
