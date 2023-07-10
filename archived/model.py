import evaluate
import collections
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    BertConfig
)
from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy

from utils import softmax
import arguments

metric = evaluate.load("squad")

args_input = arguments.get_args()
DATA_NAME = args_input.dataset_name
NUM_INIT_LB = args_input.initseed

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'
pretrain_model_dir = model_dir + '/' + DATA_NAME + '_' + str(NUM_INIT_LB) + '_' + args_input.model

def to_train(num_train_epochs, train_dataloader, device, model, optimizer, lr_scheduler, record_loss=True):
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

	# torch.save({'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()
    #             }, model_dir)
	model_to_save = model.module if hasattr(model, 'module') else model 
	model_to_save.save_pretrained(model_dir)
	print('TRAIN done!')

def get_pred(dataloader, device, features, examples, rd, record_loss=False):
    if rd == 0:
        print('get_pred use pretrain')
        config = BertConfig.from_pretrained(pretrain_model_dir, output_hidden_states=record_loss)
    else:
        print('get_pred use models')
        config = BertConfig.from_pretrained(model_dir, output_hidden_states=record_loss)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    
    # test_loss = []
    model.eval()
    start_logits = []
    end_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating_pred"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            # print('outputs:', outputs)
            # print('outputs.loss:', outputs.loss)
            # test_loss.append(outputs.loss)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    # start_logits = start_logits[: len(features)]
    # end_logits = end_logits[: len(features)]

    # if record_loss:
    #     print('test_loss:', test_loss)
    #     test_loss = sum(test_loss) / len(test_loss)
    #     # test_loss /= len(dataloader.dataset)
    #     print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

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

def logits_to_prob(start_logits, end_logits, features, batch_idx, examples, num_classes=False):
    # for num_classes=False
    prob_dict = {}
    # for num_classes=True
    probs = torch.zeros([len(batch_idx), 200])
    
    example_to_features = collections.defaultdict(list)
    max_answer_length = 30
    n_best = 20

    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append((idx, batch_idx[idx]))
    
    # for example in tqdm(examples, desc="Computing metrics"):
    for example in examples:    
        example_id = example["id"]
        answers = []
        
        # Loop through all features associated with that example
        for (feature_index, i) in example_to_features[example_id]:
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

            if num_classes:
                if 1 < len(answers) < 200: # pad to same numbers of possible answers
                    zero_list = [0] * (200 - len(answers))
                    answers.extend(zero_list)
                elif len(answers) >= 200:
                    answers = answers[:200]
                probs[feature_index] = torch.tensor(answers)
            else:
                prob_dict[i] = torch.tensor(answers)

    if num_classes:
        return probs
    else:
        return prob_dict

def get_prob(dataloader, device, features, examples, rd):
    if rd == 1:
        print('get_prob use pretrain')
        config = BertConfig.from_pretrained(pretrain_model_dir)
    else:
        print('get_pred use models')
        config = BertConfig.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    model.eval()
    prob_dict = {}
    start_logits = []
    end_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating_prob"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())
    
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    prob_dict = {}
    example_to_features = collections.defaultdict(list)
    max_answer_length = 30
    n_best = 20 # TODO: if set n_best as 5, will it effect the time??

    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    
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

            prob_dict[feature_index] = F.softmax(torch.tensor(answers), dim=0)

    return prob_dict

def get_prob_dropout(dataloader, device, features, examples, rd, n_drop=10):
    if rd == 1:
        config = BertConfig.from_pretrained(pretrain_model_dir)
    else:
        config = BertConfig.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    
    model.train()
    prob_dict = {}
    
    for i in range(n_drop):
        start_logits = []
        end_logits = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating_prob_dropout"):
                batch = {key: value.to(device) for key, value in batch.items()}
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

            if example_to_features[example_id][0] not in prob_dict:
                prob_dict[example_to_features[example_id][0]] = F.softmax(torch.tensor(answers), dim=0)
            else:
                prob_dict[example_to_features[example_id][0]] += F.softmax(torch.tensor(answers), dim=0)
            # if len(answers) > 1:
            #     if example_to_features[example_id][0] not in prob_dict:
            #         prob_dict[example_to_features[example_id][0]] = softmax(answers)
            #     else:
            #         prob_dict[example_to_features[example_id][0]] += softmax(answers)
            # elif example_to_features[example_id] != []:
            #     if example_to_features[example_id][0] not in prob_dict:
            #         prob_dict[example_to_features[example_id][0]] = np.array([0])   

    for key in prob_dict.keys():
        prob_dict[key] /= n_drop

    return prob_dict

def get_prob_dropout_split(dataloader, device, features, examples, rd, n_drop=10):
    if rd == 1:
        config = BertConfig.from_pretrained(pretrain_model_dir)
    else:
        config = BertConfig.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    model.train()
    ## use tensor to save the answers
    probs = torch.zeros([n_drop, len(dataloader.dataset), 200])
    
    for i in range(n_drop):
        start_logits = []
        end_logits = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating_prob_dropout"):
                batch = {key: value.to(device) for key, value in batch.items()}
                
                outputs = model(**batch)

                # # matually create features batch
                # data_len_batch = len(outputs.start_logits)
                # idxs_end = idxs_start + data_len_batch
                # batch_idx = list(range(idxs_start, idxs_end))
                # batch_feat = features.select(batch_idx)
                # idxs_start = idxs_end

                # out = logits_to_prob(outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy(), batch_feat, batch_idx, examples, num_classes=True)
                # prob = F.softmax(out, dim=1)
                # # deepAL+: probs[i][idxs] += F.softmax(out, dim=1).cpu()
                # probs[i][batch_idx] += F.softmax(out, dim=1).cpu()
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

                    probs[i][feature_index] += F.softmax(torch.tensor(answers), dim=0)
    return probs

def get_embeddings(dataloader, device, rd):
    if rd == 1:
        config = BertConfig.from_pretrained(pretrain_model_dir, output_hidden_states=True)
    else: 
        config = BertConfig.from_pretrained(model_dir, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)

    model.eval()
    embeddings = torch.zeros([len(dataloader.dataset), model.config.to_dict()['hidden_size']])
    idxs_start = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating_prob"):
            batch = {key: value.to(device) for key, value in batch.items()}
        
            outputs = model(**batch)
            # print('len_output:', len(outputs)) # 4
            # print('outputs:', outputs) # (loss, start_logits, end_logits, hidden_states)

            hidden_states = outputs.hidden_states
            # print('len_hidden_states:', len(hidden_states)) # 13 # each one has: (batch_size, sequence_length, hidden_size)
            # # hidden_states[0] -> last hidden states
            # print('len_hidden_states[0]:', len(hidden_states[0])) # 8, 8, 4
            # print('len_hidden_states[0][0]:', len(hidden_states[0][0])) # 384, 384, 384 # tokens in each sequence
            # print('len_hidden_states[0][0][0]:', len(hidden_states[0][0][0])) # 768, 768, 768 # number of hidden units
            # print('hidden_states:', hidden_states) 

            embedding_of_last_layer = hidden_states[0][:, 1, :] # [:, 0, :] -> to get [cls], but all the same
            # print(embedding_of_last_layer[0][0])
            idxs_end = idxs_start + len(hidden_states[0])
            embeddings[idxs_start:idxs_end] = embedding_of_last_layer.cpu()
            idxs_start = idxs_end
        
    return embeddings

def get_grad_embeddings(dataloader, device, features, examples, rd):
    if rd == 1:
        config = BertConfig.from_pretrained(pretrain_model_dir, output_hidden_states=True)
    else: 
        config = BertConfig.from_pretrained(model_dir, output_hidden_states=True)
    model = AutoModelForQuestionAnswering.from_config(config).to(device)
    model.eval()

    # deepAL+: nLab = self.params['num_class']
    nLab = 200
    embDim = model.config.to_dict()['hidden_size']
    embeddings = np.zeros([len(dataloader.dataset), embDim * nLab])

    prob_dict = []
    idxs_start = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating_prob"):
            batch = {key: Variable(value.to(device)) for key, value in batch.items()}
                
            # deepAL+: out, e1 = self.clf(x)
            outputs = model(**batch)
            # deepAL+: e1 = e1.data.cpu().numpy()
            hidden_states = outputs.hidden_states
            embedding_of_last_layer = hidden_states[0][:, 1, :]
            embedding_of_last_layer = embedding_of_last_layer.data.cpu().numpy()

            # matually create features batch
            data_len_batch = len(outputs.start_logits)
            idxs_end = idxs_start + data_len_batch
            batch_idx = list(range(idxs_start, idxs_end))
            batch_feat = features.select(batch_idx)
            idxs_start = idxs_end

            # deepAL+: batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
            # deepAL+: maxInds = np.argmax(batchProbs, 1)
            out = logits_to_prob(outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy(), batch_feat, batch_idx, examples, num_classes=True)
            batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
            maxInds = np.argmax(batchProbs, 1)

            for j in range(data_len_batch):
                for c in range(nLab):
                    if c == maxInds[j]:
                        embeddings[batch_idx[j]][embDim * c : embDim * (c+1)] = deepcopy(embedding_of_last_layer[j]) * (1 - batchProbs[j][c]) * -1.0
                    else:
                        embeddings[batch_idx[j]][embDim * c : embDim * (c+1)] = deepcopy(embedding_of_last_layer[j]) * (-1 * batchProbs[j][c]) * -1.0
            
    return embeddings