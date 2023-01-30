from transformers import MBartConfig, MBartForConditionalGeneration
from itertools import combinations
from collections import OrderedDict
from typing import List

import torch.nn as nn
import json
import argparse

"""
0,1,2,3,4,5,6,7,8,9,10,11 => 3, 7, 12 
To Leverage All Knowledge
"""

def start(args) -> nn.Module:
    distill_config = make_config(args)
    teacher = MBartForConditionalGeneration.from_pretrained(args.pretrained_model)
    student = MBartForConditionalGeneration(distill_config)

    encoder_teacher_layers = [i for i in range(teacher.config.encoder_layers)]
    decoder_teacher_layers = [i for i in range(teacher.config.decoder_layers)]

    student_encoder_layer, student_decoder_layer = make_layer(
        teacher.config,
        encoder_teacher_layers, decoder_teacher_layers, 
        args.num_encoder, args.num_decoder, mode="default"
    )
    
    model = make_student_model(
        teacher, student,
        except_encoder_layers=student_encoder_layer,
        except_decoder_layer=student_decoder_layer,
    )

    return model


def make_student_model(teacher, student: nn.Module, except_encoder_layers, except_decoder_layer) -> nn.Module:
    teacher_state_dict = teacher.state_dict()
    student_state_dict = OrderedDict()
    i = 0
    first = True
    module_flag = True
    for k, v in teacher_state_dict.items():
        if check(k, except_encoder_layers, except_decoder_layer):
            continue

        if module_flag and "decoder" in k:
            i = -1
            module_flag = False

        try:
            if k[21:22].isnumeric():
                if k[22:23].isnumeric():
                    new = k[21:23]
                else:
                    new = k[21:22]
                if first:
                    previous = new
                    first = False
                if new != previous:
                    i += 1
                if k[22:23].isnumeric():  # 10, 11
                    k = k[:21] + f"{i}" + k[23:]
                else:
                    k = k[:21] + f"{i}" + k[22:]
                previous = new
        except:
            continue
        student_state_dict[k] = v
    
    student.load_state_dict(student_state_dict)
    return student


def make_config(args) -> json:
    base_model_config = MBartConfig.from_pretrained(args.pretrained_model)
    base_model_config.encoder_layers = args.num_encoder
    base_model_config.decoder_layers = args.num_decoder
    distill_config = base_model_config

    return distill_config


def check(k: List[str], execept_encoder_layer: List[str], except_decoder_layer: List[str]):
    tmp = '.'.join(k.split('.')[1:4])
    for except_layer in execept_encoder_layer:
        if except_layer == tmp:  
            return True
    for except_layer in except_decoder_layer:
        if except_layer == tmp:
            return True
    return False


def make_layer(teacher_config, encoder_teacher_layers, decoder_teacher_layers,
               n_encoder_target: int, n_decoder_target: int, mode: str = "default"):
    en_change = False
    de_change = False
    enc_space_limit = 0
    dec_space_limit = 0
    if n_encoder_target > 6:
        en_change = True
        n_encoder_target = teacher_config.encoder_layers - n_encoder_target

    if n_decoder_target > 6:
        de_change = True
        n_decoder_target = teacher_config.decoder_layers - n_decoder_target

    if n_encoder_target != 0:
        enc_space_limit = teacher_config.encoder_layers // n_encoder_target
    if n_decoder_target != 0:
        dec_space_limit = teacher_config.decoder_layers // n_decoder_target

    tmp_encoder_distill_layers = []
    tmp_decoder_distill_layers = []

    if mode == "default":
        enc_cnd_layers = combinations(encoder_teacher_layers, n_encoder_target)
        dec_cnd_layers = combinations(decoder_teacher_layers, n_decoder_target)

        for layers in enc_cnd_layers:
            # print('enc_layer', layers)
            for idx in range(1, len(layers)):
                if abs(layers[idx - 1] - layers[idx]) < enc_space_limit:
                    break
            else:
                tmp_encoder_distill_layers.append(layers)
        # print('tmp_encoder_distill_layers pre', tmp_encoder_distill_layers)
        # [(0, 4, 8), (0, 4, 9), (0, 4, 10), (0, 4, 11), (0, 5, 9), (0, 5, 10), (0, 5, 11), (0, 6, 10), (0, 6, 11), (0, 7, 11), 
        #  (1, 5, 9), (1, 5, 10), (1, 5, 11), (1, 6, 10), (1, 6, 11), (1, 7, 11), (2, 6, 10), (2, 6, 11), (2, 7, 11), (3, 7, 11)]
        tmp_encoder_distill_layers = tmp_encoder_distill_layers[0]

        for layers in dec_cnd_layers:
            # print('dec_layer', layers)
            for idx in range(1, len(layers)):
                if abs(layers[idx - 1] - layers[idx]) < dec_space_limit:
                    break
            else:
                tmp_decoder_distill_layers.append(layers)
        # print('tmp_decoder_distill_layers pre', tmp_decoder_distill_layers)
        # [(0, 4, 8), (0, 4, 9), (0, 4, 10), (0, 4, 11), (0, 5, 9), (0, 5, 10), (0, 5, 11), (0, 6, 10), (0, 6, 11), (0, 7, 11), 
        #  (1, 5, 9), (1, 5, 10), (1, 5, 11), (1, 6, 10), (1, 6, 11), (1, 7, 11), (2, 6, 10), (2, 6, 11), (2, 7, 11), (3, 7, 11)]
        tmp_decoder_distill_layers = tmp_decoder_distill_layers[0]

    elif mode == "start":
        tmp_encoder_distill_layers = encoder_teacher_layers[:n_encoder_target]
        tmp_decoder_distill_layers = decoder_teacher_layers[:n_decoder_target]

    elif mode == "end":
        tmp_encoder_distill_layers = encoder_teacher_layers[teacher_config.encoder_layers - n_encoder_target:]
        tmp_decoder_distill_layers = decoder_teacher_layers[teacher_config.decoder_layers - n_decoder_target:]

    else:
        raise ValueError("mode must be one of start, end, or default.")

    encoder_distill_layers = tmp_encoder_distill_layers
    decoder_distill_layers = tmp_decoder_distill_layers
    
    if mode == "default" and en_change:
        encoder_distill_layers = list(set(encoder_teacher_layers) - set(tmp_encoder_distill_layers))
    if mode == "default" and de_change:
        decoder_distill_layers = list(set(decoder_teacher_layers) - set(tmp_decoder_distill_layers))

    encoder_distill_layers = list(set(encoder_teacher_layers) - set(encoder_distill_layers))
    decoder_distill_layers = list(set(decoder_teacher_layers) - set(decoder_distill_layers))

    final_enc_list = []
    final_dec_list = []
    for encoder_layer in encoder_distill_layers:
        final_enc_list.append(f"encoder.layers.{encoder_layer}")
    for decoder_layer in decoder_distill_layers:
        final_dec_list.append(f"decoder.layers.{decoder_layer}")

    return final_enc_list, final_dec_list


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_root_dir', type=str, default='/home/jiyun/distillBART_ECK/dataset')     
        parser.add_argument('--ckpt_dir', type=str, default='/home/jiyun/distillBART_ECK/checkpoints')
        parser.add_argument('--test_result_dir', type=str, default='/home/jiyun/distillBART_ECK/test_result', help='A directory to save the test results')

        parser.add_argument('--pretrained_model', type=str, default='facebook/mbart-large-50', choices=['facebook/mbart-large-50'])
        parser.add_argument('--task', type=str, default='en-en', choices=['ko-ko', 'en-en', 'zh-zh', 'ko-en', ])

        parser.add_argument('--distilation', type=bool, default=False, help='Whether to distil the model through SFT')
        parser.add_argument('--num_encoder', type=int, default=6)
        parser.add_argument('--num_decoder', type=int, default=3)

        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        
        args = parser.parse_args()
        return args       

    args = get_args()
    model = start(args)
    print(model)
    