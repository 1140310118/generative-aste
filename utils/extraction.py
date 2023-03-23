import os
import random
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl 
from transformers import AutoTokenizer

from . import load_json, load_line_json, tgenerate_batch


sep1 = ' | '
sep2 = ' ; '
lite_sep1 = '|'
lite_sep2 = ';'


_sentiment_to_word = {
    'POS': 'positive',
    'NEU': 'neutral' ,
    'NEG': 'negative',
    'positive': 'POS',
    'neutral' : 'NEU',
    'negative': 'NEG',
}
def sentiment_to_word(key):
    if key not in _sentiment_to_word:
        return 'UNK'
    return _sentiment_to_word[key]



class DataCollator:
    def __init__(self, tokenizer, max_seq_length, mode):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode

    def tok(self, text, max_seq_length):
        kwargs = {
            'text': text,
            'return_tensors': 'pt'
        }

        if max_seq_length in (-1, 'longest'):
            kwargs['padding'] = True

        else:
            kwargs['max_length'] = self.max_seq_length
            kwargs['padding'] = 'max_length'
            kwargs['truncation'] = True

        batch_encodings = self.tokenizer(**kwargs)
        return batch_encodings    

    def __call__(self, examples):
        IDs  = [example['ID'] for example in examples]
        text = [example['sentence'] for example in examples]

        batch_encodings = self.tok(text, self.max_seq_length)
        input_ids = batch_encodings['input_ids']
        attention_mask = batch_encodings['attention_mask']

        labels = None
        if self.mode in ('train', 'dev', 'test'):
            labels = self.make_labels(examples)

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'ID'            : IDs,
            'labels'        : labels,
        }

    def make_labels(self, examples):
        triplets_seqs = []
        for i in range(len(examples)):
            triplets_seq = self.make_triplets_seq(examples[i])
            triplets_seqs.append(triplets_seq)

        batch_encodings = self.tok(triplets_seqs, -1)
        labels = batch_encodings['input_ids']
        labels = torch.tensor([
            [(l if l != self.tokenizer.pad_token_id else -100)
             for l in label]
            for label in labels
        ])

        return labels

    def make_triplets_seq(self, example):
        if 'triplets_seq' in example:
            return example['triplets_seq']

        return make_triplets_seq(example)



def make_triplets_seq(example):
    triplets_seq = []
    for triplet in sorted(
        example['triplets'],
        key=lambda t: (t['aspect'][0], t['opinion'][0])
    ):  
        triplet_seq = (
            triplet['aspect'][-1] + 
            sep1 + 
            triplet['opinion'][-1] + 
            sep1 + 
            sentiment_to_word(triplet['sentiment'])
        )
        triplets_seq.append(triplet_seq)

    return sep2.join(triplets_seq)



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str='',
        max_seq_length: int = -1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        data_dir: str = '',
        dataset: str = '',
        seed: int = 42,
    ):

        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.max_seq_length     = max_seq_length
        self.train_batch_size   = train_batch_size
        self.eval_batch_size    = eval_batch_size
        self.seed               = seed

        if dataset != '':
            self.data_dir       = os.path.join(data_dir, dataset)
        else:
            self.data_dir       = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def load_dataset(self):
        train_file_name = os.path.join(self.data_dir, 'train.json')
        dev_file_name   = os.path.join(self.data_dir, 'dev.json')
        test_file_name  = os.path.join(self.data_dir, 'test.json')

        self.raw_datasets = {
            'train': load_json(train_file_name),
            'dev'  : load_json(dev_file_name),
            'test' : load_json(test_file_name)
        }
        print('-----------data statistic-------------')
        print('Train', len(self.raw_datasets['train']))
        print('Dev',   len(self.raw_datasets['dev']))
        print('Test',  len(self.raw_datasets['test']))

    def load_predict_dataset(self, version):
        if version == 'v1':
            self.load_predict_dataset_v1()
        elif version == 'v2':
            self.load_predict_dataset_v2()

    def load_predict_dataset_v1(self):
        examples = load_json(self.data_dir)
        for example in examples:
            example['triplets_seq'] = make_triplets_seq(example)

        self.raw_datasets = {'predict': examples}

        print('-----------data statistic-------------')
        print('Predict', len(self.raw_datasets['predict']))

    def load_predict_dataset_v2(self, max_example_num=20_000):

        import re 
        import spacy 
        from tqdm import tqdm 

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('sentencizer')

        min_length = 20
        max_length = 400

        dataset = list(load_line_json(self.data_dir))

        predict_examples = []
        for batch_examples in tgenerate_batch(dataset, bz=32):

            texts = [example['Text'] for example in batch_examples]
            docs  = nlp.pipe(texts, disable=['tagger', 'tok2vec', 'parser', 'lemmatizer', 'ner'])

            for doc, example in zip(docs, batch_examples):
                for i, sentence in enumerate(doc.sents):
                    sentence = str(sentence).strip()
                    sentence = sentence.replace('\r', '')
                    # '(good)' -> '( good )'
                    sentence = re.sub(r'\((?P<v1>[^ ])(?P<v2>.*)(?P<v3>[^ ])\)', lambda x: '( ' + x.group('v1') + x.group('v2') + x.group('v3') + ' )', sentence)

                    if not (min_length <= len(sentence) <= max_length):
                        continue 

                    new_example = {
                        'ID': f"{example['ID']}-{i+1}",
                        'sentence': sentence
                    }
                    predict_examples.append(new_example)
                    if len(predict_examples) >= max_example_num:
                        break

        self.raw_datasets = {'predict': predict_examples}

        print('-----------data statistic-------------')
        print('Predict', len(self.raw_datasets['predict']))

    def get_dataloader(self, mode, batch_size, shuffle):
        dataloader = DataLoader(
            dataset=self.raw_datasets[mode],
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=8,
            num_workers=1,
            collate_fn=DataCollator(
                tokenizer=self.tokenizer, 
                max_seq_length=self.max_seq_length,
                mode=mode
            )
        )

        print('dataloader-'+mode, len(dataloader))
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train', self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("dev", self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.eval_batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.get_dataloader("predict", self.eval_batch_size, shuffle=False)



class F1_Measure:
    def __init__(self):
        self.pred_list = []
        self.true_list = []

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_list.append((idx, pred))
            
    def true_inc(self, idx, trues):
        for true in trues:
            self.true_list.append((idx, true))
            
    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_list, self.true_list)
        return self.f1
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_list, true_list):
        n_tp = 0
        for pred in pred_list:
            if pred in true_list:
                n_tp += 1    
        _p = n_tp / len(pred_list) if pred_list else 1
    
        n_tp = 0
        for true in true_list:
            if true in pred_list:
                n_tp += 1 
        _r = n_tp / len(true_list) if true_list else 1

        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0

        return f1, _p, _r



def parse_triplet(triplet_seq, example):
    if triplet_seq.count(lite_sep1) != 2:
        return False

    aspect, opinion, sentiment = triplet_seq.split(lite_sep1)
    aspect  = aspect.strip()
    opinion = opinion.strip()
    sentiment = sentiment_to_word(sentiment.strip())

    if aspect not in example['sentence']:
        return False

    if opinion not in example['sentence']:
        return False

    if sentiment == 'UNK':
        return False

    return aspect, opinion, sentiment



class Result:
    def __init__(self, data):
        self.data = data 

    def __ge__(self, other):
        return self.monitor >= other.monitor

    def __gt__(self, other):
        return self.monitor >  other.monitor

    @classmethod
    def parse_from(cls, outputs, examples):
        data = {}
        examples = {example['ID']: example for example in examples}

        for output in outputs:
            IDs = output['ID']
            predictions = output['predictions']

            for ID in IDs:
                if ID not in data:
                    example = examples[ID]
                    sentence = example['sentence']
                    data[ID] = {
                        'ID': ID,
                        'sentence': sentence,
                        'triplets': example['triplets'],
                        'triplet_preds' : [],
                    }

            for ID, prediction in zip(IDs, predictions):
                example = data[ID]
                triplet_seqs = prediction.split(lite_sep2)
                for triplet_seq in triplet_seqs:
                    triplet = parse_triplet(triplet_seq, example)
                    if not triplet:
                        continue

                    example['triplet_preds'].append(triplet)

        return cls(data)

    def cal_metric(self):
        f1 = F1_Measure()

        for ID in self.data:
            example = self.data[ID]
            g = [(t['aspect'][-1], t['opinion'][-1], t['sentiment']) 
                  for t in example['triplets']]
            p = example['triplet_preds']
            f1.true_inc(ID, g)
            f1.pred_inc(ID, p)

        f1.report()

        self.detailed_metrics = {
            'f1': f1['f1'],
            'recall': f1['r'],
            'precision': f1['p'],
        }

        self.monitor = self.detailed_metrics['f1']

    def report(self):
        for metric_names in (('precision', 'recall', 'f1'),):
            for metric_name in metric_names:
                value = self.detailed_metrics[metric_name] if metric_name in self.detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()
