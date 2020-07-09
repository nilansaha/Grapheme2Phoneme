### Grapheme to Phoneme Model

A Seq2Seq model has been trained on the [cmudict dataset](https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict) for this task.

#### Usage

##### Download the dataset

```
curl -o cmudict.dict.txt 'https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict'
```

##### Training

```
python3 train.py -h
usage: train.py [-h] [--data_file DATA_FILE] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --data_file DATA_FILE
  --device DEVICE
```
example:

```
python3 train.py --data_file cmudict.dict.txt --device cpu
```

##### Inference

```
python3 infer.py -h
usage: infer.py [-h] [--pretrained_model PRETRAINED_MODEL]
                [--pronunce_vocab PRONUNCE_VOCAB] [--char_vocab CHAR_VOCAB]
                [--word WORD]

optional arguments:
  -h, --help            show this help message and exit
  --pretrained_model PRETRAINED_MODEL
  --pronunce_vocab PRONUNCE_VOCAB
  --char_vocab CHAR_VOCAB
  --word WORD
```
example:
```
python3 infer.py --pretrained_model trained_model.bin --pronunce_vocab pronunce_vocab.bin --char_vocab char_vocab.bin --word 'hello'
```

Output : 
```
HH EH1 L OW0
```
