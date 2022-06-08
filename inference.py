import torch
import argparse
import sentencepiece as spm
from sacremoses import MosesTokenizer, MosesDetokenizer
from utils.util import Config
from utils.model import load_model
from models.transformer.module import create_src_mask, create_trg_mask




def infer_seq(model, tokenizer, config, max_tokens=100):
    with torch.no_grad():
        print('Type "quit" to terminate Inference')
        while True:
            seq = input('\nUser Input Sentence >> ')
            if seq == 'quit':
                print(' --- Terminate the Service! ---')
                print('------------------------------------')
                break
            
            #Tokenize user Input with Moses
            moses_tokenizer = MosesTokenizer(lang='en')
            moses_detokenizer = MosesDetokenizer(lang='de')
            src = moses_tokenizer.tokenize(src)

            #Convert tokens to ids with sentencepiece vocab
            src = tokenizer.EncodeAsIds(seq)
            if config.model == 'seq2seq':
                src[1:-1] = seq[-2:0:-1]

            #Convert ids to tensor
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0)

            src = model.embedding(src)
            enc_out = model.encoder(src)
            trg_indice = [tokenizer.bos_id()]

            for _ in range(max_tokens):
                trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
                trg = model.embedding(trg)

                dec_out, _ = model.decoder()
                out = model.fc_out(dec_out)

                pred_token = out.argmax(2)[:, -1].item()
                trg_indice.append(pred_token)

                if pred_token == tokenizer.eos_id():
                    break
            
            pred_seq = trg_indice[1:]
            pred_seq = tokenizer.Decode(pred_seq)
            pred_seq = moses_detokenizer.detokenize(pred.split())

            print(f"Model Output Sentence >> {pred_seq}")




def infer_trans(model, tokenizer, config, max_tokens=100):
    with torch.no_grad():
        print('Type "quit" to terminate Model Inference')
        while True:
            seq = input('\nUser Input sentence >> ')
            if seq == 'quit':
                print(' --- Terminate the Survice! ---')
                break
            
            #Tokenize user Input with Moses
            moses_tokenizer = MosesTokenizer(lang='en')
            moses_detokenizer = MosesDetokenizer(lang='en')
            src = moses_tokenizer.tokenize(src)

            #Convert tokens to ids with sentencepiece vocab
            src = tokenizer.EncodeAsIds(seq)

            #Convert ids to tensor
            src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(config.device)
            
            src_mask = create_src_mask(src)
            src = model.embedding(src)
            enc_out = model.encoder(src, src_mask)
            trg_indice = [tokenizer.bos_id()]

            for _ in range(max_tokens):
                trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0).to(config.device)
                trg_mask = create_trg_mask(trg_tensor)

                trg = model.embedding(trg_tensor)

                dec_out, _ = model.decoder(enc_out, trg, src_mask, trg_mask)
                out = model.fc_out(dec_out)

                pred_token = out.argmax(2)[:, -1].item()
                trg_indice.append(pred_token)

                if pred_token == tokenizer.eos_id():
                    break
                
            pred_seq = trg_indice[1:]
            pred_seq = tokenizer.Decode(pred_seq)
            pred_seq = moses_detokenizer.detokenize(pred.split())

            print(f"Model Output Sentence >> {pred_seq}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-task', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    args = parser.parse_args()
    

    assert args.model in ['seq2seq', 'attention', 'transformer']
    assert args.task in ['translation', 'dialogue']

    config = Config(args)
    config.device = torch.device('cpu')

    #Load Model
    model = load_model(config)
    model_state = torch.load(f'checkpoints/{config.task}/{config.model}_states.pt', map_location=config.device)['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()

    #Load Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(f'data/{config.task}/vocab/spm.model')
    tokenizer.SetEncodeExtraOptions('bos:eos')


    if config.model == 'transformer':
        infer_trans(model, tokenizer, config)
    else:
        infer_seq(model, tokenizer, config)