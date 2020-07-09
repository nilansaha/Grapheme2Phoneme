import torch
import pickle
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from validation import validate
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Seq2Seq
from dataset_utils import load_data, create_vocab, G2PDataset

def main(data_file, device):
    train_data, val_data = load_data(data_file)
    char_vocab, pronunce_vocab = create_vocab(train_data)

    train_dataset = G2PDataset(train_data, char_vocab, pronunce_vocab)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    val_dataset = G2PDataset(val_data, char_vocab, pronunce_vocab)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

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

    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = Adam(model.parameters(), lr = LEARNING_RT)
    TRG_PAD_IDX = pronunce_vocab['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    EPOCHS = 5
    
    print('Training Model')
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (word, target) in enumerate(tqdm(train_loader)):
            word = word.permute(1,0)
            target = target.permute(1,0)
            word, target = word.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(word, target)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            target = target[1:].contiguous().view(-1)

            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        norm_loss = round(epoch_loss/len(train_loader), 3)
        val_loss, val_wer = validate(model, criterion, val_loader)
        print(f'EPOCH - {epoch + 1} | Train Loss - {norm_loss} | Val Loss - {val_loss} | Val WER - {val_wer}')
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'trained_model.bin')

    with open('char_vocab.bin', 'wb') as f:
        pickle.dump(char_vocab, f)

    with open('pronunce_vocab.bin', 'wb') as f:
        pickle.dump(pronunce_vocab, f)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='cmudict.dict.txt', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    args = parser.parse_args()

    main(args.data_file, args.device)
