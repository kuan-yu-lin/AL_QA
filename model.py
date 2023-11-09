import evaluate
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering
from torch.cuda import amp
import os
from collections import Counter
import string
import re
import arguments

metric = evaluate.load("squad")

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
DATA_NAME = args_input.dataset
STRATEGY_NAME = args_input.ALstrategy
MODEL_NAME = args_input.model
LOW_RES = args_input.low_resource
EXP_ID = str(args_input.exp_id)

if args_input.dev_mode:
	MODEL_DIR = os.path.abspath('') + '/dev_models'
else:
	MODEL_DIR = os.path.abspath('') + '/models'

strategy_model_dir = MODEL_DIR + '/' + EXP_ID
pretrain_model_dir =  os.path.abspath('') + '/pretrain_models' + '/' + MODEL_NAME + '_SQuAD_full_dataset_lr_3e-5'

def to_train(num_train_epochs, train_dataloader, device, model, optimizer, lr_scheduler, record_loss=False):
	if LOW_RES:
		print('Training was performed using {} query data, i.e. {} data.'.format(NUM_QUERY, len(train_dataloader.dataset)))
	else:
		print('Training was performed using the sum of {} initial data and {} query data, i.e. {} data.'.format(NUM_INIT_LB, NUM_QUERY, len(train_dataloader.dataset)))
	
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
	model_to_save.save_pretrained(strategy_model_dir)
	print('TRAIN done!')

def to_pretrain(num_train_epochs, train_dataloader, device, model, optimizer, lr_scheduler, scaler):
	print('Training was performed using the full dataset ({} data).'.format(len(train_dataloader.dataset)))
	for epoch in range(num_train_epochs):
		model.train()
		for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
			batch = {key: value.to(device) for key, value in batch.items()}
			with amp.autocast():
				outputs = model(**batch)
				loss = outputs.loss
			
			scaler.scale(loss).backward()

			scaler.step(optimizer)
			scaler.update()
			lr_scheduler.step()
			optimizer.zero_grad()

		print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

	model_to_save = model.module if hasattr(model, 'module') else model 
	model_to_save.save_pretrained(pretrain_model_dir)
	print('TRAIN done!')

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = defaultdict(list)
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

def compute_metrics_lowRes(start_logits, end_logits, features, examples):
    example_to_features = defaultdict(list)
    max_answer_length = 30
    n_best = 20
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = dict()
    for example in tqdm(examples, desc="Computing metrics"):
        example_id = example["qid"]
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
            predicted_answers[example_id] = best_answer["text"]
        else:
            predicted_answers[example_id] = ""

    theoretical_answers = dict()
    for ex in examples: theoretical_answers[ex["qid"]] = ex["answers"]
    return evaluation(theoretical_answers, predicted_answers)

def get_pred(dataloader, device, features, examples):
    model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
    
    model.eval()
    start_logits = []
    end_logits = []

    for batch in tqdm(dataloader, desc="Evaluating_pred"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(features)]
    end_logits = end_logits[: len(features)]

    if LOW_RES:
        return compute_metrics_lowRes(start_logits, end_logits, features, examples)
    else:
        return compute_metrics(start_logits, end_logits, features, examples)

def get_pretrain_pred(dataloader, device, features, examples):
    model = AutoModelForQuestionAnswering.from_pretrained(pretrain_model_dir).to(device)
    
    model.eval()
    start_logits = []
    end_logits = []

    for batch in tqdm(dataloader, desc="Evaluating_pred"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(features)]
    end_logits = end_logits[: len(features)]

    return compute_metrics(start_logits, end_logits, features, examples)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluation(theoretical_answers, predicted_answers, skip_no_answer=False):
    '''
	theoretical_answers, datatype=dict
	{strings of id: list of ground truth answers}
	predicted_answers, datatype=dict
	{strings of id: strings of prediction text}
	'''
    f1 = exact_match = total = 0
    for qid, ground_truths in theoretical_answers.items():
        if qid not in predicted_answers:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predicted_answers[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}