import math
import time
import argparse

import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score
import sentencepiece as spm

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import eval_epoch
from utils.util import Config, epoch_time, set_seed




def get_bleu(model, dataloader, tokenizer, config):
    model.eval()
    candidates, references = [], []

    for i, batch in enumerate(dataloader):
        src, trg = batch[0].to(config.device), batch[1].to(config.device)
        
        with torch.no_grad():        
            if config.model == 'transformer':
                pred = model(src, trg[:, :-1])
            else:
                pred = model(src, trg)
        
        pred = pred.argmax(-1)
        pred = pred.tolist()
        trg = trg[:, 1:].tolist()

        for can, ref in zip(pred, trg):
            can = tokenizer.Decode(can).split()
            ref = tokenizer.Decode(ref).split()
        
            candidates.append(can)
            references.append([ref])


    score = bleu_score(candidates, references, weights=[0.25, 0.25, 0.25, 0.25])
    
    return round(score * 100, 2)




def run(config):
    chk_file = f"checkpoints/{config.task}/{config.model}_states.pt"
    test_dataloader = get_dataloader('test', config)

    #Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{config.task}/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')

    #Load Model
    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.task}/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)


    print('Test')
    start_time = time.time()
    test_loss = eval_epoch(model, test_dataloader, criterion, config)
    
    if config.task == 'translate':
        test_bleu = get_bleu(model, test_dataloader, tokenizer, config)

    end_time = time.time()
    test_mins, test_secs = epoch_time(start_time, end_time)

    if config.task == 'translate':
        print(f"[ Test Loss: {test_loss} / Test BLEU Score: {test_bleu} / Time: {test_mins}min {test_secs}sec ]")
    elif config.task == 'dialogue':
        print(f"[ Test Loss: {test_loss} / Test PPL Score: {math.exp(test_loss):.2f} / Time: {test_mins}min {test_secs}sec ]")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-task', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    args = parser.parse_args()

    assert args.model in ['seq2seq', 'attention', 'transformer']
    assert args.task in ['translate', 'dialogue']
 
    set_seed()
    config = Config(args)
    run(config)