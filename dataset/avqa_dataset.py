import ast
import torch
import os
import json
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class AVQA_dataset(Dataset):

    def __init__(self, config, split, logger: logging):
        self.config = config
        self.logger = logger
        self.split = split  # train/val/test : string
        self.samples = self.load_split_data()
        ans_quelen = self.get_ans2ix_max_que_len()
        self.answer_to_ix = ans_quelen['ans2ix']
        self.max_que_len = ans_quelen['max_que_len']
        self.config['hyper_para']['num_labels'] = len(self.answer_to_ix)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") # bert-base-uncased

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question = self.samples[index]['question_content']
        ques_type = ast.literal_eval(self.samples[index]['type'])

        labels = torch.tensor(data=[self.answer_to_ix[sample['anser']]], dtype=torch.long)
        inputs = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_que_len,
            add_special_tokens=True
        )

        inputs['input_ids'] = inputs['input_ids'].squeeze()  # squeeze: delete dim when dim=1
        inputs['token_type_ids'] = inputs['token_type_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()

        position_ids = torch.arange(start=0, end=self.max_que_len, dtype=torch.long)
        name = sample['video_id']
        audio_emb = np.load(os.path.join(self.config['path']['audio_feat'], name + '.npy'))
        video_emb = np.load(os.path.join(self.config['path']['video_feat'], name + '.npy'))
        if self.config['hyper_para']['sample']:  # sample or not
            audio_emb = audio_emb[::6, :]
            audio_emb = torch.from_numpy(audio_emb)
            audio_token_type_ids = torch.ones(audio_emb.shape[0], dtype=torch.long)

            video_emb = video_emb[::6, :]
            video_emb = torch.from_numpy(video_emb)
            video_token_type_ids = torch.ones(video_emb.shape[0], dtype=torch.long)
        else:
            audio_emb = torch.from_numpy(audio_emb)
            audio_token_type_ids = torch.ones(audio_emb.shape[0], dtype=torch.long)

            video_emb = torch.from_numpy(video_emb)
            video_token_type_ids = torch.ones(video_emb.shape[0], dtype=torch.long)
        return {
            'inputs': inputs,  # tokenizer output
            'pos': position_ids,  # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])
            'audio_emb': audio_emb,
            'audio_token_type': audio_token_type_ids,  # tensor([1, 1, ... , 1, 1, 1, 1])
            'video_emb': video_emb,
            'video_token_type': video_token_type_ids,
            'labels': labels,  # tensor([ans_inx])
            'type': ques_type #---------------
        }

    # load split data json file
    def load_split_data(self):
        if self.split in 'train':
            with open(file=self.config['path']['train_data'], mode='r') as f:
                samples = json.load(f)
                samples = self.proc_que_in_split(samples)
        elif self.split in 'val':
            with open(file=self.config['path']['val_data']) as f:
                samples = json.load(f)
                samples = self.proc_que_in_split(samples)
        else: # test
            with open(file=self.config['path']['test_data'][self.split]) as f:
                samples = json.load(f)
                samples = self.proc_que_in_split(samples)
        return samples

    # load the indices of answers and the maximum lengths of questions
    def get_ans2ix_max_que_len(self):
        if os.path.exists(self.config['path']['ans_quelen']):
            with open(file=self.config['path']['ans_quelen'], mode='r') as f:
                ans_quelen = json.load(f)
        else:
            ans_quelen = {}
            ans2ix = {}
            answer_index = 0
            max_que_len = 0

            # statistic answer in train split
            with open(file=self.config['path']['train_data'], mode='r') as f:
                samples = json.load(f)
            for sample in tqdm(samples):
                que_len = len(sample['question_content'].lstrip().rstrip().split(' '))
                if max_que_len < que_len:
                    max_que_len = que_len

                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1

            # statistic answer in val split
            with open(file=self.config['path']['val_data'], mode='r') as f:
                samples = json.load(f)
            for sample in samples:
                que_len = len(sample['question_content'].rstrip().split(' '))
                if max_que_len < que_len:
                    max_que_len = que_len

                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1

            # store it to a dict ,then to a json file
            with open(file=self.config['path']['ans_quelen'], mode='w') as f:
                ans_quelen['ans2ix'] = ans2ix
                ans_quelen['max_que_len'] = max_que_len
                json.dump(obj=ans_quelen, fp=f)
        return ans_quelen

    # replace <templ> to template value in question string
    def proc_que_in_split(self, samples):
        for index, sample in enumerate(samples):
            question = sample['question_content'].lstrip().rstrip().split(' ')
            question[-1] = question[-1][:-1]  # delete '?'

            templ_value_index = 0
            for word_index in range(len(question)):
                if '<' in question[word_index]:
                    question[word_index] = ast.literal_eval(sample['templ_values'])[templ_value_index]
                    templ_value_index = templ_value_index + 1
            samples[index]['question_content'] = ' '.join(question)  # word list -> question string
        return samples
