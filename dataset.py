from torch.utils.data import Dataset, DataLoader
from transformers import MBart50Tokenizer

from pdb import set_trace
import math
import os
import pandas as pd
import re
import argparse

from utils import get_lang_code


class PawsxDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.task = args.task
        data_root_dir = args.data_root_dir
        langs, src_lang_code, tgt_lang_code = get_lang_code(args)
        self.langs = langs
        self.tokenizer = args.tokenizer
            
        assert mode in ['train', 'val', 'test']
        assert langs[0] == langs[1], '[!] src-tgt should be the same | For only paraphrasing'  # should add some codes for translation
        for lang in langs:
            if mode == 'train':
                if lang == 'en':
                    file_name = 'train'
                else:
                    file_name = 'translated_train'
            elif mode == 'val':
                file_name = 'dev_2k'
            else:
                file_name = 'test_2k'

        self.data_dir = os.path.join(data_root_dir, lang, file_name+'.tsv')    
        with open(self.data_dir) as f:
            lines = f.readlines()

        self.dataset = []
        for i, line in enumerate(lines[1:]):
            line = line.replace('\n', '')
            items = line.split('\t')
            assert len(items) == 4
            if self.langs[0] == self.langs[1] and int(items[-1]) == 0:
                continue
            self.dataset.append(items)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", self.dataset[idx][1])
        label_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", self.dataset[idx][2])
        if self.langs[0] == self.langs[1]:
            return (input_text, label_text)
    
    # not used
    def _get_max_len(self, batch):
        lens = []
        for i, data_pair in enumerate(batch):
            lens.append(len(data_pair.input_ids))
            lens.append(len(data_pair.labels))
        max_len = max(lens)

        return max_len

    def collate_fn(self, batch):
        inputs, labels = [], []
        for i, data in enumerate(batch):
            inputs.append(data[0])
            labels.append(data[1])
        data = self.tokenizer(inputs, text_target=labels, return_tensors='pt', padding='longest')
        
        return data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', default='/home/jiyun/distillBART_ECK/dataset')     
    parser.add_argument('--task', default='ko-ko')
    
    args = parser.parse_args()
    args.tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang='ko_KR', tgt_lang='ko_KR')
    
    
    dataset = PawsxDataset(args, 'train')
    print(len(dataset))
    print(dataset[201])
    exit()

    dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        print(batch)
        
        exit()
