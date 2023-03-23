import os
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl 
pl.seed_everything(42)


from pytorch_lightning.callbacks import BasePredictionWriter
from train import LightningModule as _LightningModule

from utils.extraction import DataModule, lite_sep2
from utils import save_line_json, save_json



class LightningModule(_LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__(hparams, data_module)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        num_beams = 4
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_return_sequences=num_beams,
            num_beams=num_beams,
        )

        generateds = self.data_module.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        generateds_beam = []
        for i in range(len(generateds)//num_beams):
            generateds_beam.append(generateds[i*num_beams:i*num_beams+num_beams])

        return {
            'ID': batch['ID'],
            'predictions': generateds_beam
        }    

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')
        
        parser.add_argument("--dataset_version", type=str)

        return parser




class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, predict_examples, write_interval='epoch'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.predict_examples = predict_examples

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        examples = self.predict_examples
        examples = {example['ID']: example for example in examples}

        output_examples = []
        for output in tqdm(predictions[0]):
            IDs = output['ID']
            predictions = output['predictions']

            for ID, prediction in zip(IDs, predictions):
                example  = examples[ID]
                sentence = example['sentence']

                new_example = {
                    'ID': ID,
                    'sentence': example['sentence'],
                    'triplets_prd': prediction,
                }
                if 'triplets_seq' in example:
                    new_example['triplets_seq'] = example['triplets_seq']

                output_examples.append(new_example)

        print(f'save {len(output_examples)} to', self.output_dir)
        if len(output_examples) > 10_000:
            save_line_json(output_examples, self.output_dir)
        else:
            save_json(output_examples, self.output_dir)

        if 'triplets_seq' in output_examples[0]:
            self.cal_metric(output_examples)

    def cal_metric(self, output_examples):
        n_precision = 0
        prec_hit    = 0
        n_recall    = 0
        recall_hit  = 0

        best_n_precision = 0
        best_prec_hit    = 0
        best_n_recall    = 0
        best_recall_hit  = 0

        beam_indices = [0, 0, 0, 0, 0]
        for example in output_examples:
            preds = example['triplets_prd'] + ['']
            true  = example['triplets_seq']

            _, np, ph, nr, rh = self._cal_metric(preds[0], true)
            n_precision += np
            prec_hit    += ph
            n_recall    += nr
            recall_hit  += rh

            _, np, ph, nr, rh, i = max([self._cal_metric(pred, true)+(i, ) for i, pred in enumerate(preds)])
            beam_indices[i] += 1
            best_n_precision += np
            best_prec_hit    += ph
            best_n_recall    += nr
            best_recall_hit  += rh
            
            if i != 0:
                print(example['sentence'])
                for j, pred in enumerate(preds):
                    print((j if j != i else '√'), pred)
                print()
                print('t', true)
                print()

        precision, recall, f1 = self._cal_f1(prec_hit, n_precision, recall_hit, n_recall)
        best_precision, best_recall, best_f1 = self._cal_f1(best_prec_hit, best_n_precision, best_recall_hit, best_n_recall)
        
        detailed_metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_f1': best_f1
        }
        for metric_names in (('precision', 'recall', 'f1'),('best_precision', 'best_recall', 'best_f1')):
            for metric_name in metric_names:
                value = detailed_metrics[metric_name] if metric_name in detailed_metrics else 0
                print(f'{metric_name}: {value:.4f}', end=' | ')
            print()

        print('beam_indices', beam_indices)

    def _cal_metric(self, pred, true):
        split = lambda string: [s.strip() for s in string.split(lite_sep2)]

        pred = split(pred)
        true = split(true)

        n_precision = len(pred)
        prec_hit    = 0
        n_recall    = len(true)
        recall_hit  = 0

        for p in pred:
            if p in true:
                prec_hit += 1

        for t in true:
            if t in pred:
                recall_hit += 1

        precision, recall, f1 = self._cal_f1(prec_hit, n_precision, recall_hit, n_recall)

        return (f1, -n_precision), n_precision, prec_hit, n_recall, recall_hit

    def _cal_f1(self, prec_hit, n_precision, recall_hit, n_recall):
        precision = prec_hit / n_precision if n_precision else 1
        recall    = recall_hit / n_recall if n_recall else 1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        return precision, recall, f1






def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    data_module = DataModule.from_argparse_args(args)
    data_module.load_predict_dataset(args.dataset_version)

    model = LightningModule(args, data_module)

    pred_writer = CustomWriter(
        output_dir=os.path.join(args.output_dir, 'data', args.output_sub_dir), 
        predict_examples=data_module.raw_datasets['predict']
    )
    kwargs = {
        'callbacks': [pred_writer],
        'logger': False,
        'enable_checkpointing': False,
    }
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)

    predictions = trainer.predict(
        model, 
        datamodule=data_module, 
        return_predictions=False
    )



if __name__ == '__main__':
    main()