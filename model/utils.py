import datetime
import torch
import numpy as np

class LOG:
    def __init__(self, arg):
        current_time = datetime.datetime.now()
        description = "[{}][{}][{}][{}][{}][{}]".format(
            arg.dataset,
            "unseen " + str(arg.unseen),
            arg.rel_split_seed,
            arg.lr,
            "bs " + str(arg.batch_size),
            "margin:" + str(arg.margin)
        )
        self.file_name = "../" + "[" + current_time.strftime("%m %d") + "]" + description + ".log"

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            current_time = datetime.datetime.now()
            current_time = "[" + current_time.strftime("%m/%d_%H:%M") + "]"
            with open(self.file_name, 'a+') as f_log:
                f_log.write(current_time + s + '\n')


def calculate_similarity(query_vector, target_vectors):
    similarity = torch.cosine_similarity(query_vector, target_vectors, dim=-1)
    return similarity


def weight_mean(sentence_vectors):
    def cosine_similarity_(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        similarity = dot_product / (norm_v1 * norm_v2)
        return similarity

    num_sentences = len(sentence_vectors)

    similarity_matrix = np.zeros((num_sentences, num_sentences))
    for i in range(num_sentences):
        for j in range(num_sentences):
            similarity_matrix[i, j] = cosine_similarity_(sentence_vectors[i], sentence_vectors[j])

    sentence_weights = np.mean(similarity_matrix, axis=1)

    merged_vector = np.average(sentence_vectors, axis=0, weights=sentence_weights)

    return merged_vector
