import evaluate
import collections
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    BertConfig
)

from utils import softmax
import arguments

metric = evaluate.load("squad")

args_input = arguments.get_args()
DATA_NAME = args_input.dataset_name
NUM_INIT_LB = args_input.initseed

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'
pretrain_model_dir = model_dir + '/' + DATA_NAME + '_' + str(NUM_INIT_LB) + '_' + args_input.model

def to_train(num_train_epochs, train_dataloader, device, model, optimizer, lr_scheduler, record_loss=False):
	print('Num of train dataset:', len(train_dataloader.dataset))
	for epoch in range(num_train_epochs):
		model.train()
		for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
			batch = {key: value.to(device) for key, value in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()

			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()

		if record_loss:
			print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

	model_to_save = model.module if hasattr(model, 'module') else model 
	model_to_save.save_pretrained(model_dir)
	print('TRAIN done!')

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    max_answer_length = 30
    n_best = 20
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples, desc="Computing metrics"):
        example_id = example["id"]
        context = example["context"]
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

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def get_pred(eval_dataloader, device, features, examples, record_loss=False, rd_0=False):
    if rd_0:
        config = BertConfig.from_pretrained(pretrain_model_dir, output_hidden_states=True)
    else:
        config = BertConfig.from_pretrained(model_dir, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    
    test_loss = []
    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(eval_dataloader, desc="Evaluating_pred"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            print(outputs)
            print(outputs.loss)
            test_loss.append(outputs.loss)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(features)]
    end_logits = end_logits[: len(features)]

    if record_loss:
        test_loss /= len(eval_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return compute_metrics(start_logits, end_logits, features, examples)

def get_prob(eval_dataloader, device, features, examples):
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)

    model.eval()
    start_logits = []
    end_logits = []

    for batch in tqdm(eval_dataloader, desc="Evaluating_prob"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(features)]
    end_logits = end_logits[: len(features)]

    prob_dict = {}
    example_to_features = collections.defaultdict(list)
    max_answer_length = 30
    n_best = 20 # TODO: if set n_best as 5, will it effect the time??
    
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

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
                prob_dict[feature_index] = softmax(answers)
            elif example_to_features[example_id] != []:
                prob_dict[feature_index] = np.array([0])
    
    return prob_dict

def get_prob_dropout(eval_dataloader, device, features, examples, n_drop=10):
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
    
    model.train()
    prob_dict = {}
    
    for i in range(n_drop):
        start_logits = []
        end_logits = []
        for batch in tqdm(eval_dataloader, desc="Evaluating_prob_dropout"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(features)]
        end_logits = end_logits[: len(features)]

        example_to_features = collections.defaultdict(list)
        max_answer_length = 30
        n_best = 20
            
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        n = 0
        for example in tqdm(examples):
            example_id = example["id"]
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

            if 1 < len(answers) < 200: # pad to same numbers of possible answers
                zero_list = [0] * (200 - len(answers))
                answers.extend(zero_list)
            elif len(answers) >= 200:
                answers = answers[:200]

            if len(answers) > 1:
                if example_to_features[example_id][0] not in prob_dict:
                    prob_dict[example_to_features[example_id][0]] = softmax(answers)
                else:
                    prob_dict[example_to_features[example_id][0]] += softmax(answers)
            elif example_to_features[example_id] != []:
                if example_to_features[example_id][0] not in prob_dict:
                    prob_dict[example_to_features[example_id][0]] = np.array([0])   

    for key in prob_dict.keys():
        prob_dict[key] /= n_drop

    return prob_dict

def get_prob_dropout_split(eval_dataloader, device, features, examples, n_drop=10):
    ## use tensor to save the answers

    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
    model.train()

    probs = torch.zeros([n_drop, len(eval_dataloader.dataset), 200])
    
    for i in range(n_drop):
        start_logits = []
        end_logits = []
        for batch in tqdm(eval_dataloader, desc="Evaluating_prob_dropout"):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(features)]
        end_logits = end_logits[: len(features)]

        example_to_features = collections.defaultdict(list)
        max_answer_length = 30
        n_best = 20
            
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        n = 0
        for example in tqdm(examples, desc="Computing metrics"):
            example_id = example["id"]
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

            
                if 1 < len(answers) < 200: # pad to same numbers of possible answers
                    zero_list = [0] * (200 - len(answers))
                    answers.extend(zero_list)
                elif len(answers) >= 200:
                    answers = answers[:200]

                probs[i][feature_index] += torch.tensor(softmax(answers))

    return probs
