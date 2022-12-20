import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class Encoder(tf.keras.Model):
    def __init__(self, inp_vocab_size, embedding_dim, lstm_size, input_length):
        super().__init__()

        seed = 42
        self.inp_vocab_size = inp_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.embedding = Embedding(
            input_dim=self.inp_vocab_size,
            output_dim=self.embedding_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0, stddev=1, seed=seed
            ),
            input_length=self.input_length,
            mask_zero=True,
            name="Encoder_Embedding",
        )
        self.lstm1 = LSTM(
            self.lstm_size,
            return_state=True,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.orthogonal(seed=seed),
            name="Encoder_LSTM1",
        )
        self.lstm2 = LSTM(
            self.lstm_size,
            return_state=True,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.orthogonal(seed=seed),
            name="Encoder_LSTM2",
        )

    def call(self, input):
        input_sequence, states = input[0], input[1]
        input_embedded = self.embedding(input_sequence)
        self.enc_output, self.last_hidden_state, self.last_current_state = self.lstm1(
            input_embedded, initial_state=states
        )
        self.enc_output, self.last_hidden_state, self.last_current_state = self.lstm2(
            self.enc_output, [self.last_hidden_state, self.last_current_state]
        )

        return self.enc_output, self.last_hidden_state, self.last_current_state

    def initialize_states(self, batch_size):
        # Initialized with tf.zeros
        self.first_hidden_state, self.first_current_state = tf.zeros(
            [batch_size, self.lstm_size]
        ), tf.zeros([batch_size, self.lstm_size])
        return self.first_hidden_state, self.first_current_state


class Attention(tf.keras.Model):
    def __init__(self, lstm_size, scoring_function):
        super(Attention, self).__init__()
        self.lstm_size = lstm_size
        self.scoring_function = scoring_function
        self.W = tf.keras.layers.Dense(lstm_size)


def call(self, input):
    decoder_hidden_state, encoder_output = input[0], input[1]
    decoder_hidden_state = tf.expand_dims(decoder_hidden_state, axis=2)
    output = self.W(encoder_output)
    score = tf.keras.layers.Dot(axes=(2, 1))([output, decoder_hidden_state])
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = tf.reduce_sum(attention_weights * encoder_output, axis=1)
    return context_vector, attention_weights


class Timestep_Decoder(tf.keras.Model):
    def __init__(
        self,
        out_vocab_size,
        embedding_dim,
        input_length,
        lstm_size,
        scoring_function,
        embedding_matrix=None,
    ):
        super().__init__()
        seed = 42
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.lstm_size = lstm_size
        self.scoring_function = scoring_function
        self.attention = Attention(self.lstm_size, self.scoring_function)
        self.embedding_matrix = embedding_matrix

        if self.embedding_matrix is None:
            self.embedding = Embedding(
                input_dim=self.out_vocab_size,
                output_dim=self.embedding_dim,
                embeddings_initializer=tf.keras.initializers.RandomNormal(
                    mean=0, stddev=1, seed=seed
                ),
                input_length=self.input_length,
                mask_zero=True,
                name="embedding_layer_decoder",
            )
        else:
            self.embedding = Embedding(
                input_dim=self.out_vocab_size,
                output_dim=self.embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(
                    self.embedding_matrix
                ),
                trainable=False,
                input_length=self.input_length,
                mask_zero=True,
                name="embedding_layer_decoder",
            )
        self.lstm1 = LSTM(
            self.lstm_size,
            return_state=True,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.orthogonal(seed=seed),
            name="Timestep_Decoder_LSTM1",
        )
        self.lstm2 = LSTM(
            self.lstm_size,
            return_state=True,
            return_sequences=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.orthogonal(seed=seed),
            name="Timestep_Decoder_LSTM2",
        )
        self.dense = Dense(out_vocab_size)

    def call(self, input):
        input_token, encoder_output, encoder_hidden, encoder_current = (
            input[0],
            input[1],
            input[2],
            input[3],
        )

        embedded_token = self.embedding(input_token)
        context_vector, attention_weights = self.attention(
            [encoder_hidden, encoder_output]
        )
        query_with_time_axis = tf.expand_dims(context_vector, 1)
        out_concat = tf.concat([query_with_time_axis, embedded_token], axis=-1)
        dec_output, encoder_hidden, encoder_current = self.lstm1(
            out_concat, [encoder_hidden, encoder_current]
        )
        dec_output, encoder_hidden, encoder_current = self.lstm2(
            dec_output, [encoder_hidden, encoder_current]
        )

        out = self.dense(tf.reshape(dec_output, (-1, dec_output.shape[2])))
        return out, encoder_hidden, encoder_current


class Decoder(tf.keras.Model):
    def __init__(
        self,
        out_vocab_size,
        embedding_dim,
        input_length,
        lstm_size,
        scoring_function,
        embedding_matrix=None,
    ):
        super().__init__()

        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.lstm_size = lstm_size
        self.scoring_function = scoring_function
        self.embedding_matrix = embedding_matrix
        self.timestepdecoder = Timestep_Decoder(
            self.out_vocab_size,
            self.embedding_dim,
            self.input_length,
            self.lstm_size,
            self.scoring_function,
            self.embedding_matrix,
        )

    @tf.function
    def call(self, input):
        decoder_input, encoder_output, encoder_hidden, encoder_current = (
            input[0],
            input[1],
            input[2],
            input[3],
        )
        all_outputs = tf.TensorArray(
            tf.float32, size=tf.shape(decoder_input)[1], name="output_array"
        )

        loop = tf.shape(decoder_input)[1]
        for timestep in range(loop):
            output, encoder_hidden, encoder_current = self.timestepdecoder(
                [
                    decoder_input[:, timestep : timestep + 1],
                    encoder_output,
                    encoder_hidden,
                    encoder_current,
                ]
            )
            all_outputs = all_outputs.write(timestep, output)
        all_outputs = tf.transpose(all_outputs.stack(), [1, 0, 2])
        return all_outputs


class Attention_Based_Encoder_Decoder(tf.keras.Model):
    def __init__(
        self,
        input_length,
        inp_vocab_size,
        out_vocab_size,
        lstm_size,
        scoring_function,
        batch_size,
        embedding_dim,
        embedding_matrix=None,
    ):
        super().__init__()

        self.input_length = input_length
        self.inp_vocab_size = inp_vocab_size + 1
        self.out_vocab_size = out_vocab_size + 1
        self.lstm_size = lstm_size
        self.scoring_function = scoring_function
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix

        self.encoder = Encoder(
            inp_vocab_size=self.inp_vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_size=self.lstm_size,
            input_length=self.input_length,
        )

        self.decoder = Decoder(
            out_vocab_size=self.out_vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_size=self.lstm_size,
            scoring_function=self.scoring_function,
            input_length=self.input_length,
            embedding_matrix=self.embedding_matrix,
        )

    def call(self, data):
        enc_inp, dec_inp = data[0], data[1]
        initial_state = self.encoder.initialize_states(
            self.batch_size
        )  # Initialized Encoder state
        encoder_output, encoder_hidden, encoder_current = self.encoder(
            [enc_inp, initial_state]
        )  # Encoder
        final_output = self.decoder(
            [dec_inp, encoder_output, encoder_hidden, encoder_current]
        )  # Decoder
        return final_output
