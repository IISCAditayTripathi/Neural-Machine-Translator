# Neural-Machine-Translator

NMT is implemented in pytorch. It has the following dependencies:
- Python3
- Numpy
- Pytorch
- tqdm
- subword-nmt
- pickle
- NLTK

## Data Preprocessing
WMT14 has *4.5M* sentences with the vocabulary size of $1.5M$ for German corpus. Byte pair encoding is applied to reduce the vocabulary size.
WMT14 has 3 files: europarl-v7.de-en, commoncrawl.de-en, news-commentary-v10.de-en for German and English respectibly. Concatenate the files into a single file as follows:
```
cat europarl-v7.de-en.de commoncrawl.de-en.de ews-commentary-v10.de-en.de > german_corpus.txt
cat europarl-v7.de-en.en commoncrawl.de-en.en ews-commentary-v10.de-en.en > english_corpus.txt
```
Get the byte pair encoding as follows:

```
subword-nmt learn-bpe -s 30000 < german_corpus.txt > german_corpus_codefile
subword-nmt apply-bpe -c german_corpus_codefile < german_corpus.txt > german_corpus_tokenized.txt
```
```
subword-nmt learn-bpe -s 30000 < english_corpus.txt > english_corpus_codefile
subword-nmt apply-bpe -c english_corpus_codefile < english_corpus.txt > english_corpus_tokenized.txt
```

To get the parallel corpus ready for training, run the following command:
```
python3 prepare_dict.py
```
To prepare the English-hindi corpus, run the following command:
```
python3 create_dataset.py
```
## Neural Network training

To run the neural network training, run the following command:
```
python3 my_train_new_lstm.py --mode=train --gpu=0 --nb_epochs=50 --batch_size=50 --attention=dot --nb_layers=1 --hidden_size=256
```

To run the additive attention, run the following command:

```
python3 my_train_new_lstm_bandu.py --mode=train --gpu=0 --nb_epochs=50 --batch_size=50 --nb_layers=1 --hidden_size=256
```

To run the trainer for the *Hindi* translator, change the datapath from inside.

## Evaluation
To run the evaluation, run the following command:
 ```
python3 my_train_new_stm.py --mode=eval --gpu=0 --attention=dot --nb_layers=1 --hidden_size=256
```

To run the additive attention, run the following command:

```
python3 my_train_new_lstm_bandu.py --mode=eval --gpu=0 --nb_layers=1 --hidden_size=256
```

## Calculate BLEU score
To calculate BLEU score, run the following command:
```
python3 calculate_score.py --file1=file1.txt --file2=file2.txt
```
