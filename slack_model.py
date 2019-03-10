
"""
NN: http://deeplearning.hatenablog.com/entry/neural_machine_translation_theory
TS: https://qiita.com/icoxfog417/items/d06651db10e27220c819
解説: https://qiita.com/KojiOhki/items/45df6a18b08dfb63d4f9
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import preprocessing


"""
Sequence-to-sequence model with an attention mechanism

アーキテクチャ:
Encoder_Embeddingレイヤー: Word2vec
Encoder_Inputレイヤー
Encoder_隠れレイヤー: LSTM
Attentionファンクション: Softmax
Decoder_Embeddingレイヤー: tanh
Decoder_Inputレイヤー
Decoder_隠れレイヤー: LSTM
Decoder_Outputレイヤー: Softmax
Decoder_Generatingレイヤー
"""

class my_seq2seq(object):

    def __init__(self,
                 num_input_tokens, num_target_tokens, # 辞書サイズ
                 max_input_seq_length, max_target_seq_length, # インプット＆アウトプットする文の文字数
                 input_word2idx, target_word2idx, # { 文字: インデックス}
                 input_idx2word, target_idx2word, # { インデックス: 文字}
                 num_samples=512, use_lstm=False, # トレーニングデータの総文字数, RNNそうに使うDLアーキテクチャ（LSTM）
                 NUM_HIDDEN_UNITS, NUM_HIDDEN_LAYERS, # 隠れ層のユニット数とレイヤー数
                 buckets,
                 batch_size,
                 learning_rate, learning_rate_decay_factor,
                 forward_only=False,
                 max_gradient_norm, 
                 # size, num_layers,
                 ):

        """
        Seq2Seqを実現するニューラルネットを構築するのに必要な情報やハイパラメータたち
        1. 入力データの整形（辞書数,パディング,バケツ化）
        2. 入出力層の長さ
        3. 隠れ層の構造（単層or多層）、タイプ（LSTM, RNN）
        4. 損失関数や勾配降下最適化アルゴリズム
        5. 学習率やミニバッチサイズなどのハイパラメータ
        """
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_target_tokens
        self.max_input_seq_length = max_input_seq_length
        self.max_output_seq_length = max_target_seq_length
        self.input_word2idx = input_word2idx
        self.output_word2idx = target_word2idx
        self.input_idx2word = input_idx2word
        self.output_idx2word = target_idx2word

        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        """
        出力層の定義
        """
        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.num_input_tokens:

            with tf.device("/cpu0"):
                # tf.get_variableは、既に存在すれば取得し、なければ変数を作成する関数, 第一引数に変数の名前を指定する
                weight_matrix = tf.get_variable("proj_w", [size, self.num_target_tokens])
                # transpose word_matrix
                weight_matrix_T = tf.transpose(weight_matrix)
                # サンプル数（デフォルトでは512）がターゲット語彙サイズよりも小さい場合にのみサンプリング・ソフトマックスを構築する
                bias = tf.get_variable("proj_b", [self.num_target_tokens])
            # 重み行列とバイアス・ベクトルのペア
            # RNN セルは、バッチサイズ × target_vocab_size ではなく、バッチサイズ × size の形状のベクトルを返す
            # ロジットを取り出すために、重み行列を乗算し、バイアスを加える必要がある
            output_projection = (weight_matrix, bias)

            """
            <where>における誤差関数の定義
            """
            # inputs, labels = input_idx2word.keys(), input_idx2word.values()
            def sampled_loss(inputs, labels):

                with tf.device("/cpu0"):
                    #
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(weight_matrix_T, bias, labels,
                                                      inputs, num_samples, self.num_target_tokens)

            softmax_loss_function = sampled_loss

        """
        隠れ層のアーキテクチャ設定: 「隠れ層のセルの種類」と「隠れ層の数」の定義!
        """
        single_cell = rnn_cell.GRUCell(NUM_HIDDEN_UNITS)
        if use_lstm:
            single_cell = rnn_cell.BasicLSTMCell(NUM_HIDDEN_UNITS)
        # 隠れ層のユニット数の定義
        cell = single_cell
        if NUM_HIDDEN_LAYERS > 1:
            cell = rnn_cell.MultiRNNCell([single_cell] * NUM_HIDDEN_LAYERS)

        """
        Integrate each part of Neural Network Aechitecture!
        """
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return seq2seq.embedding_attention_seq2seq(
                                encoder_inputs, decoder_inputs,
                                cell, num_input_tokens,
                                num_target_tokens, output_projection=output_projection,
                                feed_previous=do_decode)

        """
        入出力データ（Sentence）の矯正その1
        # バケッティングは、文が短いときに不必要に多くの PAD 埋めを防具ために存在する
        # 入出力長を数種類に固定 (例えば，[(5,10),(10,15),(20,25),(40,50)]) して、数パターンのバケツを用意する
        """
        # 入力レイヤーの定義
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in xrange(buckets[-1][0]):
            # tf.placeholderは、事前に変数の値を定義する必要のない、計算グラフ内の変数の容れ物
            # tf.placeholderの使い方をマスターすると、実行時に任意の値を入れてTensorFlowに計算させることができる
            # 機械学習のコードでは、主に入力層に渡す変数をtf.placeholderで定義して、実行時に学習に入力画像や情報をバッチ毎に供給するために使用する
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        for i in xrange(buckets[-1][1] + 1):
            # decoder
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            # target_weights
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        #
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        #
        if forward_only:
            self.outputs, self.losses = seq2seq.model_with_buckets(
                                            self.encoder_inputs, self.decoder_inputs,
                                            targets, self.target_weights,
                                            buckets, self.num_target_tokens,
                                            lambda x, y: seq2seq_f(encoder_inputs=x, encoder_inputs=y, do_decode=True),
                                            softmax_loss_function=softmax_loss_function)

        else:
            self.outputs, self.losses = seq2seq.model_with_buckets(
                                            self.encoder_inputs, self.decoder_inputs,
                                            targets, self.target_weights,
                                            buckets, self.target_vocab_size,
                                            lambda x, y: seq2seq_f(encoder_inputs=x, encoder_inputs=y, do_decode=False),
                                            softmax_loss_function=softmax_loss_function)


        """切れた"""

        """
        バックプロパゲーションの勾配降下の設定
        """
        params = tf.trainable_variables()
        if not forward_only:
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

      """
      学習ステップ
      """
      def step(self, session, encoder_inputs, decoder_inputs, target_weights,
               bucket_id, forward_only):

        encoder_size, decoder_size = self.buckets[bucket_id]

        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        #
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        #
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)


        if not forward_only:
          output_feed = [self.updates[bucket_id],
                         self.gradient_norms[bucket_id],
                         self.losses[bucket_id]]
        else:
          output_feed = [self.losses[bucket_id]]
          for l in xrange(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
          return outputs[1], outputs[2], None
        else:
          return None, outputs[0], outputs[1:]


    """
    入出力データ（Sentence）の矯正その2
    # ミニバッチ学習では入出力長を揃えなければならない
    # パディングはあらかじめモデルの入出力長を固定値 (例えば，入力長: 5, 出力長: 10) に定め，入出力文は空白を PAD で埋める．
    # 入力文は反転して入力した方が精度が良くなる
    """

    """
    バッチ正規化（Batch Normalization）
    what: Deep Learningにおける各重みパラメータを上手くreparametrizationすることで、ネットワークを最適化するための方法の一つ
    Why: 1. 学習スピードが早くなる　2. 過学習が抑えられる
    How: 各ユニットの出力をminibatchごとにnormalizeした新たな値で置き直すことで、内部の変数の分布(内部共変量シフト)が大きく変わるのを防ぐ
    # https://qiita.com/cfiken/items/b477c7878828ebdb0387
    """

    def get_batch(self, data, bucket_id):
        # 任意のbucketを取り出す
        encoder_size, decoder_size = self.buckets[bucket_id]
        # 全入力文章をpaddingして返す
        encoder_inputs = []
        decoder_inputs = []

        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            #
            encoder_pad = [preprocess.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            #
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([preprocess.GO_ID] + decoder_input + [preprocess.PAD_ID] * decoder_pad_size)

        # ミニバッチ正規化されたembedded wordsや、重みを返す
        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_weights = []

        # encoderの単語数
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        # decoderの単語数
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        # batch weight
        batch_weight = np.ones(self.batch_size, dtype=np.float32)

        for batch_idx in xrange(self.batch_size):
            #
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            #
            if length_idx == decoder_size - 1 or target == preprocess.PAD_ID:
                batch_weight[batch_idx] = 0.0
        #
        batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
