import os
import time
import argparse
import torch
import pytorch_lightning as pl 
pl.seed_everything(42)


from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.utilities import rank_zero_only

from transformers import T5ForConditionalGeneration
from utils.extraction import DataModule
from utils.extraction import Result



class LightningModule(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
     
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path
        )

    def _make_model_dir(self):
        return os.path.join(self.hparams.output_dir, 'model', f'dataset={self.hparams.dataset},b={self.hparams.output_sub_dir},seed={self.hparams.seed}')

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = self._make_model_dir()
        print(f'## save model to {dir_name}')
        self.model.config.time = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        self.model.save_pretrained(dir_name)
        self.data_module.tokenizer.save_pretrained(dir_name)

    def load_model(self):
        dir_name = self._make_model_dir()
        print(f'## load model to {dir_name}')
        self.model = T5ForConditionalGeneration.from_pretrained(dir_name)

    def forward(self, **inputs):
        ID = inputs.pop('ID')
        output = self.model(**inputs)
    
        return {
            'ID': ID,
            'loss': output[0]
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output['loss']

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=100,
            num_return_sequences=1,
            num_beams=1,
        )

        generateds = self.data_module.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return {
            'ID': batch['ID'],
            'predictions': generateds
        }

    def validation_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['dev']
        self.current_val_result = Result.parse_from(outputs, examples)
        self.current_val_result.cal_metric()

        if not hasattr(self, 'best_val_result'):
            self.best_val_result = self.current_val_result
            self.save_model()

        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['test']
        self.test_result = Result.parse_from(outputs, examples)
        self.test_result.cal_metric()
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.hparams.warmup_steps, 
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0., type=float)

        parser.add_argument("--output_dir", type=str)
        parser.add_argument("--output_sub_dir", type=str)
        parser.add_argument("--do_train", action='store_true')

        return parser



class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print()
        if hasattr(pl_module, 'current_train_result'):
            pl_module.current_train_result.report()
        print()
        print('------------------------------------------------------------')
        print('[current]', end=' ')
        pl_module.current_val_result.report()

        print('[best]   ', end=' ')
        pl_module.best_val_result.report()
        print('------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()



def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LightningModule.add_model_specific_args(parser)
    parser = DataModule.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5

    data_module = DataModule.from_argparse_args(args)
    data_module.load_dataset()

    model = LightningModule(args, data_module)    

    logging_callback = LoggingCallback()  
    kwargs = {
        'callbacks': [logging_callback],
        'logger': False,
        'num_sanity_val_steps': 5 if args.do_train else 0,
        'enable_checkpointing': False,
    }
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    
    if args.do_train:
        trainer.fit(model, datamodule=data_module)
        model.load_model()
        trainer.test(model, datamodule=data_module)

    else:
        model.load_model()
        trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()