import torch
from torch.utils.data import DataLoader

from transformers import MBart50Tokenizer, MBartConfig, MBartForConditionalGeneration
import evaluate
from tqdm import tqdm
import os
from glob import glob

from dataset import PawsxDataset
from distill_mbart import DistillMBart
from distillation import start
from utils import get_lang_code
from train import get_args


def test(args):
    # DataLoaders
    langs, src_lang_code, tgt_lang_code = get_lang_code(args)
    tokenizer = MBart50Tokenizer.from_pretrained(args.pretrained_model, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
    args.tokenizer = tokenizer
    test_dataset = PawsxDataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Metrics
    metrics = {'bleu': evaluate.load('bleu'),
               'rouge': evaluate.load('rouge')}
    
    log = open(os.path.join(args.test_result_dir, args.test_idx_date+".txt"), 'w')
    
    # Load Model
    for idx, model_dir in enumerate(args.checkpoint_list):
        total_predictions = []
        total_references = []
        total_inputs = []
        loss_sum = 0 

        model = MBartForConditionalGeneration.from_pretrained(model_dir)
        print('[LOAD MODEL: {}]'.format(model_dir.split('/')[-1]))
        log.write('LOAD MODEL: {}\n'.format(model_dir.split('/')[-1]))

        model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_dataloader), desc='Testing CKPT {}'.format(idx), mininterval=0.01, ncols=100):
                outputs = model(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])

                loss = outputs.loss.item()
                
                predictions = outputs.logits.argmax(dim=-1)
                references = batch.labels
                inputs = tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True)
                predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                references = tokenizer.batch_decode(references, skip_special_tokens=True)

                loss_sum += loss
                total_inputs += inputs
                total_predictions += predictions
                total_references += references
        
        results = {}
        for key, metric in metrics.items():
            results[key] = metric.compute(
                predictions=total_predictions,
                references=total_references
                )

        final_loss = loss_sum / len(test_dataloader)
        
        print("CKPT: {}".format(str(model_dir)))
        print(" * loss: {}".format(final_loss))
        print(" * bleu: {}".format(results['bleu']))
        print(" * rouge: {}\n".format(results['rouge']))

        log.write("CKPT: {}\n".format(str(model_dir)))
        log.write(" * loss: {}\n".format(final_loss))
        log.write(" * bleu: {}\n".format(results['bleu']))
        log.write(" * rouge: {}\n\n".format(results['rouge']))
    
    log.close()


if __name__ == "__main__":
    args = get_args()
    args.batch_size = 1
    args.epochs = 1
    if not os.path.exists(args.test_result_dir):
        os.mkdir(args.test_result_dir)

    # args.exp_date = '0125_1532'
    args.checkpoint_list = glob('checkpoints/{}/{}*.ckpt'.format(args.test_idx_date, args.test_idx_date))
    args.checkpoint_list.sort()
    print(args.checkpoint_list)
    
    test(args)
    
