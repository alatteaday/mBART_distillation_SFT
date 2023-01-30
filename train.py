import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration

import argparse
import gc
from datetime import datetime
import os
import sys

from dataset import PawsxDataset
from utils import get_lang_code
from distill_mbart import DistillMBart


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default='x-final')     
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--test_result_dir', type=str, default='test_result', help='A directory to save the test results')

    parser.add_argument('--pretrained_model', type=str, default='facebook/mbart-large-50', choices=['facebook/mbart-large-50'])
    parser.add_argument('--task', type=str, default='en-en', choices=['ko-ko', 'en-en', 'ko-en', ])

    parser.add_argument('--distillation', type=bool, default=True, help='Whether to distil the model through SFT')
    parser.add_argument('--num_encoder', type=int, default=3)
    parser.add_argument('--num_decoder', type=int, default=9)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('-testid', '--test_idx_date', type=str, required='--test' in sys.argv)

    args = parser.parse_args()
    return args    
        

def train(args):
    # DataLoaders
    langs, src_lang_code, tgt_lang_code = get_lang_code(args)
    tokenizer = MBart50Tokenizer.from_pretrained(args.pretrained_model, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
    args.tokenizer = tokenizer
    train_dataset = PawsxDataset(args, 'train')
    val_dataset = PawsxDataset(args, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    # Load Model in PL 
    model = DistillMBart(args)
    #trainer = pl.Trainer(accelerator="gpu", devices="1,2,3", strategy="ddp")
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    gc.collect()
    torch.cuda.empty_cache()

    now = datetime.now()
    args.exp_date = now.strftime('%m%d_%H%M')
    exp_dir = os.path.join(args.ckpt_dir, args.exp_date)
    if not os.path.exists(exp_dir):
        os.mkdir(os.path.join(args.ckpt_dir, args.exp_date))
    args.exp_dir = exp_dir
   
    train(args)