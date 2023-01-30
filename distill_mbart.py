import torch
import pytorch_lightning as pl
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR

import logging
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import evaluate
import os

from distillation import start


class DistillMBart(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.epoch = -1
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_encoder = args.num_encoder
        self.num_decoder = args.num_decoder

        self.tokenizer = args.tokenizer
        
        if args.distillation:        
            print("[*] Compressed model")
            self.model = start(args)
        else:  # the original pretrained model
            print("[*] Whole teacher model")
            self.model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model)
        
        self.metrics = {'bleu': evaluate.load('bleu'),
                        'rouge': evaluate.load('rouge')}
        

    def forward(self, batch):
        # for key, v in batch.items():
        #     batch[key] = batch[key].to('cuda:3')
        # self.model = self.model.to('cuda:3')
        outputs = self.model(input_ids=batch['input_ids'], 
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'])

        return outputs

    def training_step(self, train_batch, batch_idx):
        outputs = self.forward(train_batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        outputs = self.forward(val_batch)
        loss = outputs["loss"]
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        predictions = outputs.logits.argmax(dim=-1)
        references = val_batch.labels
        predictions = self.tokenizer.batch_decode(predictions)
        references = self.tokenizer.batch_decode(references)

        return {'val_loss': loss, 'predictions': predictions, 'references': references}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return ({"optimizer": optimizer})

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_epoch_end(self, training_step_outputs):
        tr_avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        print('\n[Epoch {}] training_loss: {}\n'.format(self.epoch, tr_avg_loss))

    def validation_epoch_end(self, val_step_outputs):
        predictions = []
        references = []
        for x in val_step_outputs:
            predictions += x['predictions']
            references += x['references']
        
        results = {}
        for key, metric in self.metrics.items():
            results[key] = metric.compute(
                predictions=predictions,
                references=references
                )

        val_avg_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
        
        print("\n[Epoch {}]".format(self.epoch)) 
        print(" * validation_loss: {}".format(val_avg_loss))
        print(" * bleu: {}".format(results['bleu']))
        print(" * rouge: {}\n".format(results['rouge']))

        if self.args.distillation:
            file_name = '{}_{}-{}_{}_epoch{}_bs{}_lr{}.ckpt'.format(self.args.exp_date, self.args.num_encoder, self.args.num_decoder, self.args.task, self.epoch, self.args.batch_size, self.args.lr)   
        else:
            file_name = '{}_nodist_{}_epoch{}_bs{}_lr{}.ckpt'.format(self.args.exp_date, self.args.task, self.epoch, self.args.batch_size, self.args.lr)   
        
        self.model.save_pretrained(save_directory=os.path.join(self.args.exp_dir, file_name))
        self.epoch += 1

        if self.epoch > self.args.epochs:
            print("[!] early stopping")
            exit()

        
    
    
    

