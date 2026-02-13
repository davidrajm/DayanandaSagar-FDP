import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

text = """The quick brown fox jumps over the lazy dog The quick brown fox is very fast"""
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # Adding 1 for padding


input_sequences = []
for line in text.split("."):  # Splitting sentences (for real datasets, use full text)
    token_list = tokenizer.texts_to_sequences([line])[0]  # Convert words to numbers
    for i in range(1, len(token_list)):  
        n_gram_sequence = token_list[:i+1]  # Creating n-gram sequences
        input_sequences.append(n_gram_sequence)


# Padding sequences to make them the same length
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]  # Splitting into inputs and labels
y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # Convert to categorical

from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):  # Explicit training param
        # Fix: Pass query, key, value explicitly
        attn_output = self.att(query=inputs, key=inputs, value=inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    
embed_dim = 64  # Word embedding size
num_heads = 2   # Number of attention heads
ff_dim = 128    # Hidden layer size

input_shape = X.shape[1]
inputs = tf.keras.layers.Input(shape=(input_shape,))

embedding_layer = Embedding(total_words, embed_dim, input_length=input_shape)(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer)
flatten = tf.keras.layers.Flatten()(transformer_block)
output = Dense(total_words, activation="softmax")(flatten)
model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X, y, epochs=50, verbose=1)


def predict_next_word(seed_text, tokenizer, max_seq_length, model):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
    predicted_probs = model.predict(token_list)
    predicted_word_index = np.argmax(predicted_probs)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""


# Example Usage:
seed_text = "The quick brown"
next_word = predict_next_word(seed_text, tokenizer, max_seq_length, model)
print(f"Predicted next word: {next_word}")