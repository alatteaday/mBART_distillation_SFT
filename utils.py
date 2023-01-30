import torch


def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')


def get_lang_code(args):
    lang_list = ['ko', 'en', 'zh']
    lang_code_list = ['ko_KR', 'en_XX', 'zh_CN']

    langs = args.task.split('-')
    src_lang_code = lang_code_list[lang_list.index(langs[0])]
    tgt_lang_code = lang_code_list[lang_list.index(langs[1])]

    return langs, src_lang_code, tgt_lang_code


def save_checkpoint(args, model):
    ckpt_dir = os.path.join(args.data_root_dir, 'checkpoint')
    mkdir(ckpt_dir)

    args.waiting += 1

    args.accelerator.save_state(os.path.join(ckpt_dir, args.checkpoint))

    if args.val_losses[-1] <= min(args.val_losses):
        args.waiting = 0
        filename = 'BEST_' + args.checkpoint + '.ckpt'

        unwrapped_model = args.accelerator.unwrap_model(model)
        args.accelerator.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, filename))
        print('\t[!] The best checkpoint is updated.')

    
