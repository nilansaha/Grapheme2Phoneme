from nltk.metrics import edit_distance

def wer(gold, predicted):
    return edit_distance(gold, predicted)/len(gold)

def calculate_wer_batch(output, target):
    batch_wer = 0
    batch_size = output.shape[1]
    for i in range(batch_size):
        predicted = output[:,i,:]
        predicted = predicted.argmax(dim=1)
        gold = target[:,i]
        batch_wer += wer(predicted.tolist(), gold.tolist())
    return batch_wer/batch_size

def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0
    total_wer = 0
    for i, (word, target) in enumerate(val_loader):
        word = word.permute(1,0)
        target = target.permute(1,0)
        output = model(word, target)
        total_wer += calculate_wer_batch(output, target)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        target = target[1:].contiguous().view(-1)

        loss = criterion(output, target)
        total_loss += loss.item()
    normalized_loss = round(total_loss/len(val_loader), 3)
    norm_wer = round(total_wer/len(val_loader), 3)
    model.train()
    return normalized_loss, norm_wer
