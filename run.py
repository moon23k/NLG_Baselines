import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_model,
    Trainer,
    Tester
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.search_method = args.search

        self.ckpt = f"ckpt/{self.task}/{self.model_type}_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'

        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)    
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    set_seed()
    config = Config(args)
    model = load_model(config)
    tokenizer = load_tokenizer(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    
    
    args = parser.parse_args()
    assert args.task.lower() in ['translation', 'dialogue', 'summarization']
    assert args.mode.lower() in ['train', 'test']
    assert args.model.lower() in ['rnn', 'rnn_attn', 'transformer']


    if args.mode == 'train':
        os.makedirs(f"ckpt/{args.task}", exist_ok=True)
    else:
        assert os.path.exists(f'ckpt/{args.task}/{args.model}_model.pt')

    main(args)