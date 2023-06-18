import numpy as np
import collections
from tqdm.auto import tqdm

import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch

def random_sampling_query(labeled_idxs):
    return np.random.choice(np.where(labeled_idxs==0)[0], NUM_QUERY, replace=False)

def margin_sampling_query(start_logits, end_logits, features, examples):
    # thinking the posibility of get rid of 'examples'
    margin_dict = []
    example_to_features = collections.defaultdict(list)
    max_answer_length = 30
    n_best = 20 # TODO: if set n_best as 5, will it effect the time??
    
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        # context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append(start_logit[start_index] + end_logit[end_index])
        
        if len(answers) > 1:
            answers.sort(reverse=True)
            n_best_sorted_answers = answers[:n_best]
            margin = n_best_sorted_answers[0] - n_best_sorted_answers[1]
            margin_dict.append(
                {'idx': example_to_features[example_id], 
                 'margin': margin}
            )

        else:
            margin_dict.append(
                {'idx': example_to_features[example_id], 
                 'margin': 0}
            )

    sorted_margin_dict = sorted(margin_dict, key=lambda d: d['margin'], reverse=True)

    return [margin_dict['idx'][0] for margin_dict in sorted_margin_dict[:NUM_QUERY]]
