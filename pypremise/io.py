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

from typing import List
import os
import subprocess
import pypremise
from pypremise.core import PremiseResult, Direction, Pattern, PremiseInstance


def parse_premise_result(path: str) -> List[PremiseResult]:
    results = []
    with open(path, "r") as in_file:
        for line in in_file:
            tail, counts, bias_direction, fisher, mdl_gain, _ = line.strip().split(";")
            fisher = float(fisher)
            mdl_gain = float(mdl_gain)
            clauses = []
            for clause in tail.split(" and "):
                clause_tokens = []
                for item in clause.split("-or-"):
                    clause_tokens.append(int(item))
                clauses.append(clause_tokens)
            label0_count, label1_count = counts.strip().split(":")
            label0_count = int(label0_count.strip())
            label1_count = int(label1_count.strip())
            direction = Direction.MISCLASSIFICATION if bias_direction == "0" else Direction.CORRECT
            pattern = Pattern(clauses)
            results.append(PremiseResult(pattern, direction, label0_count, label1_count, fisher, mdl_gain))
    return results


def write_dat_content(instances: List[PremiseInstance], feature_path: str, label_path: str) -> None:
    feature_file = open(feature_path, "w")
    label_file = open(label_path, "w")
    for instance in instances:
        feature_file.write(" ".join([str(f) for f in instance.features]) + "\n")
        label_file.write(f"{instance.label.value}\n")
    feature_file.flush()
    label_file.flush()


def write_embedding_file(embedding_index_to_vector, embedding_path: str, embedding_dimensionality: int,
                         max_feature_index: int) -> None:
    out_file = open(embedding_path, "w")
    for index in range(max_feature_index+1):
            if index not in embedding_index_to_vector:
                raise Exception(f"Index {index} could not be found in the embedding map embedding_index_to_vector."
                                f"The map needs to contain values for all feature indices from 0 to "
                                f"max feature index ({max_feature_index}) even if this index does not appear in the "
                                f"dataset.")
            vector = embedding_index_to_vector[index]
            if len(vector) != embedding_dimensionality:
                raise Exception(f"Length of the embedding vector is {len(vector)} but the embedding dimensionality"
                                f"was specified as {embedding_dimensionality}.")
            out_file.write(" ".join([str(v) for v in vector]) + "\n")
    out_file.flush()


def call_premise_program(feature_path: str, label_path: str, result_path: str, embedding_path: str,
                         embedding_size: int, neighbor_max_distance: int, fisher_p_value: float,
                         clause_max_overlap: float, min_overlap: float,
                         verbose: bool = True) -> None:
    # The Premise executable is stored within the PyPremise module
    module_path = os.path.dirname(pypremise.__file__)

    if verbose:
        print("Starting Premise. This might take a while.")
    try:
        process = subprocess.Popen([os.path.join(module_path, 'Premise'), feature_path, label_path, result_path,
                                    embedding_path,
                                    str(embedding_size), str(neighbor_max_distance),
                                    str(fisher_p_value), str(clause_max_overlap), str(min_overlap)],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    except Exception as e:
        raise Exception(f"Execution of Premise failed due to {e}.")
    if verbose:
        print("Premise finished. This was the output:")
        print(stdout)
    if len(stderr) > 0:
        print("Premise reported an error:")
        print(stderr)
        raise Exception(f"Premise reported an error.")


if __name__ == "__main__":
    pass
    #rs = parse_premise_result("orig_data_val_vocMinThreshold0.result")
    #for r in rs:
    #    print(r)

    #write_dat_content(instances, "test_features.dat", "test_labels.dat")
    #call_premise_program("example_biased_features.dat", "example_biased_labels.dat", "result.test", "", str(3), str(0))