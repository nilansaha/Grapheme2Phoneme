import torch
import pickle
import argparse
from model import Encoder, Decoder, Seq2Seq
from dataset_utils import load_data, G2PDataset

def format_data(word, char_vocab, pronunce_vocab, max_len = 20):
    word = word[:max_len - 2]
    word = [char_vocab['<sos>']] + [char_vocab.get(char, char_vocab['<unk>']) for char in word] + [char_vocab['<eos>']]
    padded_word = word + [char_vocab['<pad>']] * (max_len - len(word))
    padded_word = torch.tensor([padded_word])
    padded_word = padded_word.permute(1,0)
    target = [pronunce_vocab['<sos>']] * max_len
    target = torch.tensor([target])
    target = target.permute(1,0)
    return padded_word, target

def main(pretrained_model, char_vocab_file, pronunce_vocab_file, word):

    with open(char_vocab_file, 'rb') as f:
        char_vocab = pickle.load(f)

    with open(pronunce_vocab_file, 'rb') as f:
        pronunce_vocab = pickle.load(f)

    INPUT_DIM = len(char_vocab)
    OUTPUT_DIM = len(pronunce_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    N_LAYERS = 1
    LEARNING_RT = 0.001
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    device = 'cpu'
    model = Seq2Seq(enc, dec, device).to(device)
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    word_tensor, target_tensor = format_data(word, char_vocab, pronunce_vocab)
    output = model(word_tensor, target_tensor, teacher_forcing_ratio = 0.0)
    output = output[:,0,:]
    output = output.argmax(dim=1)

    inv_pronunce_vocab = {v:k for k,v in pronunce_vocab.items()}
    output = [inv_pronunce_vocab[i] for i in output.tolist()]
    pronunciation = []
    for i in output[1:]:
        if i == '<eos>':
            break
        else:
            pronunciation.append(i)
    print(' '.join(pronunciation))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default='trained_model.bin', type=str)
    parser.add_argument('--pronunce_vocab', default='pronunce_vocab.bin', type=str)
    parser.add_argument('--char_vocab', default='char_vocab.bin', type=str)
    parser.add_argument('--word', type=str)
    args = parser.parse_args()

    main(args.pretrained_model, args.char_vocab, args.pronunce_vocab, args.word)
