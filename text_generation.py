import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle

class TextGenerationAI:
    def __init__(self, model_name="text_gen_model"):
        self.model_name = model_name
        self.model = None
        self.chars = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.seq_length = 40

    def create_model(self, vocab_size):
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.seq_length, vocab_size), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model
        return model

    def prepare_data(self, text_data):
        self.chars = sorted(list(set(text_data)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        vocab_size = len(self.chars)
        print(f"Vocabulary size: {vocab_size} unique characters")

        x_data = []
        y_data = []
        for i in range(0, len(text_data) - self.seq_length):
            seq_in = text_data[i:i + self.seq_length]
            seq_out = text_data[i + self.seq_length]
            x_data.append([self.char_to_idx[char] for char in seq_in])
            y_data.append(self.char_to_idx[seq_out])

        print(f"Total sequences: {len(x_data)}")

        x = np.zeros((len(x_data), self.seq_length, vocab_size), dtype=np.bool_)
        for i, sequence in enumerate(x_data):
            for t, char_idx in enumerate(sequence):
                x[i, t, char_idx] = 1

        y = tf.keras.utils.to_categorical(y_data, num_classes=vocab_size)

        return x, y, vocab_size

    def train(self, text_data, epochs=10, batch_size=128):
        os.makedirs(f"models/{self.model_name}", exist_ok=True)

        print("Preparing training data...")
        x, y, vocab_size = self.prepare_data(text_data)

        if self.model is None:
            print(f"Creating model with vocabulary size: {vocab_size}")
            self.create_model(vocab_size)
            self.model.summary()

        with open(f"models/{self.model_name}/chars_mapping.pkl", 'wb') as f:
            pickle.dump((self.chars, self.char_to_idx, self.idx_to_char), f)

        checkpoint = ModelCheckpoint(
            f"models/{self.model_name}/model.keras",
            monitor='loss',
            save_best_only=True
        )

        print(f"Training model on {len(x)} sequences...")
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

    def load(self):
        model_path = f"models/{self.model_name}/model.keras"
        chars_path = f"models/{self.model_name}/chars_mapping.pkl"

        if not os.path.exists(model_path) or not os.path.exists(chars_path):
            print(f"Error: Model files not found at {model_path}")
            return False

        self.model = load_model(model_path)

        with open(chars_path, 'rb') as f:
            self.chars, self.char_to_idx, self.idx_to_char = pickle.load(f)
        return True

    def generate_text(self, seed_text, length=200, temperature=0.5):
        if self.model is None or self.char_to_idx is None:
            if not self.load():
                print("Error: No model loaded. Train or load a model first.")
                return ""

        if len(seed_text) < self.seq_length:
            print(f"Seed text must be at least {self.seq_length} characters. Padding...")
            seed_text = seed_text.rjust(self.seq_length)

        pattern = seed_text[-self.seq_length:]
        print(f"Generating text with seed: {pattern}")
        result = pattern

        vocab_size = len(self.chars)

        for i in range(length):
            x = np.zeros((1, self.seq_length, vocab_size))
            for t, char in enumerate(pattern):
                if char in self.char_to_idx:
                    x[0, t, self.char_to_idx[char]] = 1

            prediction = self.model.predict(x, verbose=0)[0]

            prediction = np.log(prediction) / temperature
            exp_prediction = np.exp(prediction)
            prediction = exp_prediction / np.sum(exp_prediction)

            index = np.random.choice(range(len(prediction)), p=prediction)

            result_char = self.idx_to_char[index]
            result += result_char

            pattern = pattern[1:] + result_char

            if i % 50 == 0 and i > 0:
                print(f"Generated {i}/{length} characters")

        return result