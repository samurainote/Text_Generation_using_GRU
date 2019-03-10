
import json
import os
import shutil

import numpy as np
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.layers import Activation, dot, concatenate
from keras.models import Model


""" Single-layer LSTM Model """

def single_layer_lstm(
        input_dict_size,
        output_dict_size,
        input_length=20,
        output_length=20):

    #  define length of encoder/decoder input
    encoder_input = Input(shape=(input_length, ))
    decoder_input = Input(shape=(output_length, ))

    # Encoder
    # we can add BatchNormalization, Masking
    encoder_input2embedding = Embedding(input_dim=imput_dict_size, output_dim=64, input_length=input_length, mask_zero=True)(encoder_input)
    embedding2encoder = LSTM(units=64, return_sequences=True, dropout=0.3)(encoder_input2embedding)

    # Decoder
    # we can add BatchNormalization, Masking
    decoder_input2embedding = Embedding(input_dim=output_dict_size, output_dim=64, input_length=output_length, mask_zero=True)(decoder_input)
    embedding2decoder = LSTM(units=64, return_sequences=True, dropout=0.3)(decoder_input2embedding)
    decoder2softmax = Dense(units=output_dict_size, activation="softmax")
    softmax2output = TimeDistributed(decoder2softmax)(embedding2decoder)

    # Encoder-Decoder Model
    model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


""" Attension Single-layer LSTM Model """

def attension_single_layer_lstm(
        input_dict_size,
        output_dict_size,
        input_length=20,
        output_length=20):

    #  define length of encoder/decoder input
    encoder_input = Input(shape=(input_length, ))
    decoder_input = Input(shape=(output_length, ))

    # Encoder
    # we can add BatchNormalization, Masking
    encoder_input2embedding = Embedding(input_dim=imput_dict_size, output_dim=64, input_length=input_length, mask_zero=True)(encoder_input)
    embedding2encoder = LSTM(units=64, return_sequences=True, dropout=0.3)(encoder_input2embedding)

    # Decoder
    # we can add BatchNormalization, Masking
    decoder_input2embedding = Embedding(input_dim=output_dict_size, output_dim=64, input_length=output_length, mask_zero=True)(decoder_input)
    embedding2decoder = LSTM(units=64, return_sequences=True, dropout=0.3)(decoder_input2embedding)

    # Attention
    encoder2attention = dot([embedding2decoder, embedding2encoder], axes=[2, 2])
    attention2softmax = Activation("softmax", name="attention")(encoder2attention)

    # Context
    context_vector = dot([attention2softmax, embedding2encoder], axes=[2, 1])
    context_vector_decoder = concatenate([context_vector, embedding2decoder])

    # Attention Decoder
    decoder2tanh = Dense(units=64, activation="tanh")
    tanh2context = TimeDistributed(decoder2tanh)(context_vector_decoder)
    context2softmax = Dense(units=output_dict_size, activation="softmax")
    softmax2output = TimeDistributed(context2softmax)(tanh2context)

    # Attention Encoder-Decoder Model
    model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


""" Multi-layer LSTM Model """

def multi_layer_lstm(
        input_dict_size,
        output_dict_size,
        input_length=20,
        output_length=20):

    #  define length of encoder/decoder input
    encoder_input = Input(shape=(input_length, ))
    decoder_input = Input(shape=(output_length, ))

    # Encoder
    # we can add BatchNormalization, Masking
    encoder_input2embedding = Embedding(input_dim=imput_dict_size, output_dim=64, input_length=input_length, mask_zero=True)(encoder_input)
    embedding2encoder = LSTM(units=64, return_sequences=True, dropout=0.3)(encoder_input2embedding)

    # Decoder
    # we can add BatchNormalization, Masking
    decoder_input2embedding = Embedding(input_dim=output_dict_size, output_dim=64, input_length=output_length, mask_zero=True)(decoder_input)
    embedding2decoder = LSTM(units=64, return_sequences=True, dropout=0.3)(decoder_input2embedding)
    decoder2softmax = Dense(units=output_dict_size, activation="softmax")
    softmax2output = TimeDistributed(decoder2softmax)(embedding2decoder)

    # Encoder-Decoder Model
    model = Model(inputs=[encoder_input, decoder_input], outputs=[softmax2output])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model
