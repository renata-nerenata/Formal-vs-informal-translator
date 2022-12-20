import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
import joblib, re, os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))

    def __getitem__(self, i):
        """
        Pack the input data in tuple format: ([[encoder_inp], [decoder_inp]], decoder_out)
        """
        # Tracking indices of start and stop
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # Creating data in tuple form
        batch = [
            np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)
        ]
        return tuple([[batch[0], batch[1]], batch[2]])

    def __len__(self):
        """
        Required for model.fit method to keep logs.
        """
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """
        Callback to shuffle the indices of data on each epoch.
        """
        self.indexes = np.random.permutation(self.indexes)


class PreprocessData:
    def __init__(self, data, tknizer_formal, tknizer_informal, max_len):
        self.encoder_inps = data["encoder_inp"].values
        self.decoder_inps = data["decoder_inp"].values
        self.decoder_outs = data["decoder_out"].values
        self.tknizer_informal = tknizer_informal
        self.tknizer_formal = tknizer_formal
        self.max_len = max_len

    def __getitem__(self, i):
        """
        Tokenize data, zero-pad to make all sequences of same length.
        """
        # Tokenizing the sequences by passing them in lists as required by tokenizer
        self.encoder_inp_seq = self.tknizer_formal.texts_to_sequences(
            [self.encoder_inps[i]]
        )
        self.decoder_inp_seq = self.tknizer_informal.texts_to_sequences(
            [self.decoder_inps[i]]
        )
        self.decoder_out_seq = self.tknizer_informal.texts_to_sequences(
            [self.decoder_outs[i]]
        )
        # Padding the sequences with zeros
        self.encoder_inp_seq = pad_sequences(
            self.encoder_inp_seq, maxlen=self.max_len, dtype="int32", padding="post"
        )
        self.decoder_inp_seq = pad_sequences(
            self.decoder_inp_seq, maxlen=self.max_len, dtype="int32", padding="post"
        )
        self.decoder_out_seq = pad_sequences(
            self.decoder_out_seq, maxlen=self.max_len, dtype="int32", padding="post"
        )
        return self.encoder_inp_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self):
        """
        Required for model.fit method to keep logs.
        """
        return len(self.encoder_inps)


def get_tokenized_data(train):
    tknizer_formal = Tokenizer(
        filters='"#$%&()*+-/=@[\\]^_`{|}~\t\n', lower=False, char_level=True
    )
    tknizer_informal = Tokenizer(
        filters='"#$%&()*+-/=@[\\]^_`{|}~\t\n', lower=False, char_level=True
    )

    train.iloc[0]["encoder_inp"] = str(train.iloc[0]["encoder_inp"]) + ">"
    train["decoder_inp"].iloc[0] = train["decoder_inp"].iloc[0] + ">"
    tknizer_formal.fit_on_texts(train["encoder_inp"].values)
    tknizer_informal.fit_on_texts(train["decoder_inp"].values)

    vocab_size_formal = len(tknizer_formal.word_index.keys())
    vocab_size_informal = len(tknizer_informal.word_index.keys())
