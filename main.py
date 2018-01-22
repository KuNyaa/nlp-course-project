import sys
import csv
import nltk
import datetime
import argparse
import regexDFAEquals
import numpy as np
import tensorflow as tf

#hyper-parameters
ID = ''
np.random.seed(3)
class FLAGS():
    data_path = "./data/turk/"
    vocab_freq_threshold = 3
    epochs = 90
    batch_size = 128
    num_layers = 3
    learning_rate = 0.001
    embedding_size = 256
    hidden_size = 512
    dropout_keep_prob = 0.6
    load = False
    inference = False
    
#end hyper-parameters

def get_vocab(text):
    vocab_fd = nltk.FreqDist([word for sent in text for word in sent])
    vocab = sorted(set([word for (word, freq) in vocab_fd.items() if freq >= FLAGS.vocab_freq_threshold]))
    vocab = ['<EOS>', '<SOS>', '<UNK>'] + vocab
    return vocab

def load_data(path):
    file = open(path, 'r')
    text = []
    for line in file:
        if line:
            text.append(line.strip().split())
    return text

def encode(vocab, text, reverce=False):
    text_encoded = []
    for sent in text:
        sent_encoded = []
        for word in sent:
            if word in vocab:
                sent_encoded.append(vocab.index(word))
            else:
                sent_encoded.append(vocab.index('<UNK>'))
        if reverce:
            sent_encoded.reverse()
        text_encoded.append(sent_encoded)
        
    return text_encoded

def decode(vocab, text):
    text_decoded = []
    for sent in text:
        sent_decoded = []
        for word in sent:
            sent_decoded.append(vocab[word])
        text_decoded.append(sent_decoded)
    return text_decoded


def make_dataset(path, data_src, data_tgt=False, time_major=True):
    path = "./data/temp/" + path + ID
    def save(path, data):
        with open(path + '.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def load(path):
        dataset = tf.data.TextLineDataset(path + '.csv')
        dataset = dataset.map(lambda string: tf.string_split([string], delimiter=',').values)
        dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
        return dataset

    save(path + '_src', data_src)
    if not data_tgt:
        dataset = load(path + '_src')
        dataset = dataset.map(lambda x: (x, tf.size(x)))
        dataset = dataset.padded_batch(
            FLAGS.batch_size,
            padded_shapes=([None], []),
            padding_values=(FLAGS.src_eos_id, 0))
        if time_major:
            dataset = dataset.map(lambda x, y: (tf.transpose(x), y))

        return dataset

    save(path + '_tgt', data_tgt)
    
    dataset = tf.data.Dataset.zip((load(path + '_src'), load(path + '_tgt')))
    tgt_sos_pad = tf.constant(FLAGS.tgt_sos_id, shape=[1], dtype=tf.int32)
    dataset = dataset.map(lambda x, y: (x, tf.concat([tgt_sos_pad, y], axis=0)))
    dataset = dataset.map(lambda x, y: ((x, tf.size(x)), (y, tf.size(y))))
    dataset = dataset.padded_batch(
        FLAGS.batch_size,
        padded_shapes=(([None], []), ([None], [])),
        padding_values=((FLAGS.src_eos_id, 0), (FLAGS.tgt_eos_id, 0)))

    if time_major:
        dataset = dataset.map(lambda x, y: ((tf.transpose(x[0]), x[1]), (tf.transpose(y[0]), y[1])))
        
    return dataset


class NMT(object):
    def __init__(self, src_vocab_size, tgt_vocab_size, tgt_sos_id, tgt_eos_id, embedding_size,
                 hidden_size, num_layers, dropout_keep_prob):
        
        encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_inputs')
        decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_inputs')
        encoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='encoder_lengths')
        decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='decoder_lengths')
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.encoder_lengths = encoder_lengths
        self.decoder_lengths = decoder_lengths
        self.learning_rate = learning_rate
        self.lr_summary_op = tf.summary.scalar('learning_rate', learning_rate)
        
        max_encoder_time, batch_size = tf.shape(encoder_inputs)[0], tf.shape(encoder_inputs)[1]

        tgt_eos_pad = tf.fill([1, batch_size], tgt_eos_id, name='tgt_eos_pad')
        decoder_outputs = tf.transpose(tf.concat([decoder_inputs[1:, :], tgt_eos_pad], axis=0))
        target_weights = tf.cast(tf.sequence_mask(decoder_lengths), dtype=tf.float32)
        self.decoder_outputs = decoder_outputs
        self._debug_tranweight_shape = tf.shape(target_weights)
        
        with tf.variable_scope('encoder'):
            #embedding
            embedding_encoder = tf.get_variable('embedding_encoder',
                                            [src_vocab_size, embedding_size],
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1))
            encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

            #encoder RNN
            cells = [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)]
            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            encoder_initial_state = encoder_cell.zero_state(batch_size, dtype=tf.float32)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                                               sequence_length=encoder_lengths,
                                                               initial_state=encoder_initial_state,
                                                               time_major=True)

        with tf.variable_scope('decoder'):
            #embedding
            embedding_decoder = tf.get_variable('embedding_decoder',
                                            [tgt_vocab_size, embedding_size],
                                            initializer=tf.random_uniform_initializer(-1.0, 1.0))

            #attention
            attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
            num_units = hidden_size
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, attention_states,
                                                            memory_sequence_length=encoder_lengths)
            
            #decoder RNN
            cells = [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)]
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            decoder_cell_attn = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=num_units)
            decoder_cell_attn_dropout = tf.nn.rnn_cell.DropoutWrapper(
                decoder_cell_attn,
                output_keep_prob=dropout_keep_prob)
            
            projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
            
            with tf.variable_scope('train'):
                decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)
                
                #helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, decoder_lengths,
                    time_major=True)
                #decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell_attn_dropout, helper,
                    decoder_cell_attn_dropout.zero_state(batch_size, dtype=tf.float32),
                    output_layer=projection_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = outputs.rnn_output
                self._debug_logits_shape = tf.shape(logits)
            

            with tf.variable_scope('inference'):
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding_decoder,
                    tf.fill([batch_size], tgt_sos_id),
                    tgt_eos_id)
                maximum_iterations = max_encoder_time * 5
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell_attn, helper,
                    decoder_cell_attn.zero_state(batch_size, dtype=tf.float32),
                    output_layer=projection_layer)
                outputs, _, outputs_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=False,
                    maximum_iterations=maximum_iterations)
                translations = outputs.sample_id
                self.translations = translations
                self.outputs_lengths = outputs_lengths
            
        with tf.variable_scope('loss'):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=decoder_outputs,
                logits=logits)
            train_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
            self.train_loss = train_loss
            
        with tf.variable_scope('gradients'):
            params = tf.trainable_variables()
            gradients = tf.gradients(train_loss, params)
            max_gradient_norm = 1
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.zeros_initializer(tf.int32),
                                      trainable=False)
        self.global_step = global_step
        
        with tf.variable_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            train_op = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                 global_step=global_step)
            self.train_op = train_op

        with tf.variable_scope('save-model'):
            self.saver = tf.train.Saver(max_to_keep=1)
            self.save_path = "./checkpoints/model" + ID + \
                             "--embedding-size-" + str(FLAGS.embedding_size) + \
                             "--hidden-size-" + str(FLAGS.hidden_size) + \
                             "--num_layers-" + str(FLAGS.num_layers) + \
                             "--batch_size-" + str(FLAGS.batch_size) + \
                             "--learning-rate-" + str(FLAGS.learning_rate) + \
                             "--dropout-keep-prob-" + str(FLAGS.dropout_keep_prob) + \
                             "/"
            self.max_test_acc = 0

        with tf.variable_scope('summary'):
            summary_path = "./summaries/model" + ID + \
                           "--embedding-size-" + str(FLAGS.embedding_size) + \
                           "--hidden-size-" + str(FLAGS.hidden_size) + \
                           "--num_layers-" + str(FLAGS.num_layers) + \
                           "--batch_size-" + str(FLAGS.batch_size) + \
                           "--learning-rate-" + str(FLAGS.learning_rate) + \
                           "--dropout-keep-prob-" + str(FLAGS.dropout_keep_prob) + \
                           "-summary"
            self.summary_writer = tf.summary.FileWriter(summary_path)
            self.summary_loss = tf.placeholder(dtype=tf.float32, shape=[], name='summary_loss')
            self.train_str = tf.placeholder(dtype=tf.float32, shape=[], name='train_str_equal')
            self.dev_str = tf.placeholder(dtype=tf.float32, shape=[], name='dev_str_equal')
            self.dev_DFA = tf.placeholder(dtype=tf.float32, shape=[], name='dev_DFA_equal')
            self.loss_summary_op = tf.summary.scalar('loss', self.summary_loss)
            tf.summary.scalar('train-str-equal', self.train_str)
            tf.summary.scalar('dev-str-equal', self.dev_str)
            tf.summary.scalar('dev-DFA-equal', self.dev_DFA)
            self.all_summary_op = tf.summary.merge_all()
            
    def train(self, sess, train_dataset, test_dataset, learning_rate):
        iterator = train_dataset.make_one_shot_iterator()
        next_batch_op = iterator.get_next()
        max_test_acc = 0
        while True:
            try:
                next_batch = sess.run(next_batch_op)
                ((encoder_inputs, encoder_lengths), (decoder_inputs, decoder_lengths)) = next_batch 
            except:
                break
            feed_dict = {self.encoder_inputs:encoder_inputs, self.encoder_lengths:encoder_lengths,
                         self.decoder_inputs:decoder_inputs, self.decoder_lengths:decoder_lengths,
                         self.learning_rate:learning_rate}
            _, loss, global_step = sess.run([self.train_op, self.train_loss, self.global_step],
                                            feed_dict=feed_dict)
            if global_step % 500 == 0:
                time_str = datetime.datetime.now().isoformat()
                train_loss = loss
                train_str_acc, train_DFA_acc = self.eval(sess, train_dataset)
                test_str_acc, test_DFA_acc = self.eval(sess, test_dataset, True)
                if test_DFA_acc > max_test_acc:
                    self.save(sess)
                    max_test_acc = test_DFA_acc
                print("{}: step:{}".format(time_str, global_step))
                print("train_loss:{}".format(train_loss))
                print("train_str_acc:{}  test_str_acc:{}".format(train_str_acc, test_str_acc))
                print("test_DFA_acc:{}".format(test_DFA_acc))
                feed_dict = {self.learning_rate:learning_rate, self.summary_loss:train_loss,
                             self.train_str:train_str_acc, self.dev_str:test_str_acc,
                             self.dev_DFA:test_DFA_acc}
                all_summary = sess.run(self.all_summary_op, feed_dict=feed_dict)
                self.summary_writer.add_summary(all_summary, global_step)
                
            elif global_step % 100 == 0:
                feed_dict = {self.learning_rate:learning_rate, self.summary_loss:loss}
                lr_summary, loss_summary = sess.run([self.lr_summary_op, self.loss_summary_op],
                                                    feed_dict=feed_dict)
                self.summary_writer.add_summary(lr_summary, global_step)
                self.summary_writer.add_summary(loss_summary, global_step)
                

    def eval(self, sess, dataset, eval_DFA=False):
        iterator = dataset.make_one_shot_iterator()
        next_batch_op = iterator.get_next()
        num_batch = 0
        avrg_acc = 0
        DFA_avrg_acc = 0
        while True:
            try:
                next_batch = sess.run(next_batch_op)
                ((encoder_inputs, encoder_lengths), (decoder_inputs, decoder_lengths)) = next_batch
                num_batch += 1
            except:
                break
            
            feed_dict = {self.encoder_inputs:encoder_inputs, self.encoder_lengths:encoder_lengths}
            translations, outputs_lengths = sess.run(
                [self.translations, self.outputs_lengths],
                feed_dict=feed_dict)
            batch_size = translations.shape[0]
            encoder_inputs = np.transpose(encoder_inputs)
            decoder_outputs = np.transpose(decoder_inputs)
            assert(translations.shape[0] == batch_size)

            acc, DFA_acc = 0, 0

            sample = np.random.randint(0, batch_size)
            
            for i in range(batch_size):
                src_sent_length = encoder_lengths[i]
                gold_sent_length = decoder_lengths[i]
                tran_sent_length = outputs_lengths[i]

                while tran_sent_length > 1 and translations[i, tran_sent_length - 2] == FLAGS.tgt_eos_id:
                    tran_sent_length -= 1
                
                src_sent = ' '.join(reversed(decode(FLAGS.src_vocab, [encoder_inputs[i, :src_sent_length]])[0]))     
                gold_sent = ' '.join(decode(FLAGS.tgt_vocab, [decoder_outputs[i, 1:gold_sent_length]])[0])
                tran_sent = ' '.join(decode(FLAGS.tgt_vocab, [translations[i, :tran_sent_length - 1]])[0])

                if gold_sent == tran_sent:
                    acc += 1
                
                if eval_DFA and regexDFAEquals.regex_equiv_from_raw(gold_sent, tran_sent):
                    DFA_acc += 1
                
                #sample result
                if num_batch == 1 and i == sample:
                    print('src: ' + src_sent)
                    print('gold: ' + gold_sent)
                    print('tran: ' + tran_sent)
                

            avrg_acc += acc / batch_size
            DFA_avrg_acc += DFA_acc / batch_size

        avrg_acc /= num_batch
        DFA_avrg_acc /= num_batch

        return avrg_acc, DFA_avrg_acc
    
    def infer(self, sess, dataset):
        iterator = dataset.make_one_shot_iterator()
        next_batch_op = iterator.get_next()
        result = []
        while True:
            try:
                next_batch = sess.run(next_batch_op)
                encoder_inputs, encoder_lengths = next_batch
            except:
                break
            
            feed_dict = {self.encoder_inputs:encoder_inputs, self.encoder_lengths:encoder_lengths}
            translations, outputs_lengths = sess.run([self.translations, self.outputs_lengths],
                                                     feed_dict=feed_dict)
            batch_size = translations.shape[0]
            assert(translations.shape[0] == batch_size)

            for i in range(batch_size):
                tran_sent_length = outputs_lengths[i]
                
                tran_sent = ' '.join(decode(FLAGS.tgt_vocab, [translations[i, :tran_sent_length - 1]])[0])
                tran_sent = tran_sent.replace("<VOW>", " ".join('AEIOUaeiou'))
                tran_sent = tran_sent.replace("<NUM>", " ".join('0-9'))
                tran_sent = tran_sent.replace("<LET>", " ".join('A-Za-z'))
                tran_sent = tran_sent.replace("<CAP>", " ".join('A-Z'))
                tran_sent = tran_sent.replace("<LOW>", " ".join('a-z'))
                tran_sent = tran_sent.replace(" ", "")
                result.append(tran_sent)
                
        with open(FLAGS.data_path + "result-infer.txt", 'w') as file:
            for line in result:
                file.write(line + '\n')
        

    def save(self, sess):
        global_step = sess.run(self.global_step)
        self.saved_global_setp = global_step
        self.saver.save(sess, self.save_path + 'ckpt', global_step)
        print('Model saved.')

    def load(self, sess, path):
        model_file=tf.train.latest_checkpoint(path)
        self.saver.restore(sess, model_file)
        print('Model restored.')
                

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', help="data_path", 
                          type=str, default='')
    parser.add_argument('--vocab_freq_threshold', help="vocab_freq_threshold", 
                          type=int, default=0)
    parser.add_argument('--epochs', help="epochs", 
                          type=int, default=0)
    parser.add_argument('--batch_size', help="batch_size", 
                          type=int, default=0)
    parser.add_argument('--num_layers', help="num_layers", 
                          type=int, default=0)
    parser.add_argument('--learning_rate', help="learning_rate", 
                          type=float, default=0)
    parser.add_argument('--embedding_size', help="embedding_size", 
                          type=int, default=0)
    parser.add_argument('--hidden_size', help="hidden_size", 
                          type=int, default=0)
    parser.add_argument('--dropout_keep_prob', help="dropout_keep_prob", 
                          type=float, default=0)
    parser.add_argument('--load', help="load", 
                          type=bool, default=False)
    parser.add_argument('--inference', help="inference", 
                          type=bool, default=False)
    
    args = parser.parse_args(arguments)

    if args.data_path:
        FLAGS.data_path = args.data_path
    if args.vocab_freq_threshold:
        FLAGS.vocab_freq_threshold = args.vocab_freq_threshold
    if args.epochs:
        FLAGS.epochs = args.epochs
    if args.batch_size:
        FLAGS.batch_size = args.batch_size
    if args.num_layers:
        FLAGS.num_layers = args.num_layers
    if args.learning_rate:
        FLAGS.learning_rate = args.learning_rate
    if args.embedding_size:
        FLAGS.embedding_size = args.embedding_size
    if args.hidden_size:
        FLAGS.hidden_size = args.hidden_size
    if args.dropout_keep_prob:
        FLAGS.dropout_keep_prob = args.dropout_keep_prob
    if args.load:
        FLAGS.load = args.load
    if args.inference:
        FLAGS.inference = args.inference


    train_src_raw = load_data(FLAGS.data_path + 'src-train.txt')
    train_tgt_raw = load_data(FLAGS.data_path + 'targ-train.txt')
    dev_src_raw = load_data(FLAGS.data_path + 'src-val.txt')
    dev_tgt_raw = load_data(FLAGS.data_path + 'targ-val.txt')
    test_src_raw = load_data(FLAGS.data_path + 'src-test.txt')
    test_tgt_raw = load_data(FLAGS.data_path + 'targ-test.txt')
    
    src_vocab = get_vocab(train_src_raw)
    tgt_vocab = get_vocab(train_tgt_raw)
    FLAGS.src_vocab = src_vocab
    FLAGS.tgt_vocab = tgt_vocab
    FLAGS.src_vocab_size = len(src_vocab)
    FLAGS.tgt_vocab_size = len(tgt_vocab)
    FLAGS.src_eos_id = src_vocab.index('<EOS>')
    FLAGS.tgt_sos_id = tgt_vocab.index('<SOS>')
    FLAGS.tgt_eos_id = tgt_vocab.index('<EOS>')
    
    train_src, train_tgt = encode(src_vocab, train_src_raw, True), encode(tgt_vocab, train_tgt_raw)
    dev_src, dev_tgt = encode(src_vocab, dev_src_raw, True), encode(tgt_vocab, dev_tgt_raw)
    test_src, test_tgt = encode(src_vocab, test_src_raw, True), encode(tgt_vocab, test_tgt_raw)
    
    
    train_dataset = make_dataset('train', train_src, train_tgt)
    dev_dataset = make_dataset('dev', dev_src, dev_tgt)
    test_dataset = make_dataset('test', test_src, test_tgt)

    
        
    nmt = NMT(FLAGS.src_vocab_size, FLAGS.tgt_vocab_size, FLAGS.tgt_sos_id, FLAGS.tgt_eos_id,
              FLAGS.embedding_size, FLAGS.hidden_size, FLAGS.num_layers, FLAGS.dropout_keep_prob)

    config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if FLAGS.load:
        nmt.load(sess, nmt.save_path)
    else :
        
        lr_decay_threshold = 1 * FLAGS.epochs // 3
        lr_decay_rate = FLAGS.learning_rate / (FLAGS.epochs - lr_decay_threshold)
        learning_rate = FLAGS.learning_rate
        
        for k in range(FLAGS.epochs):
            if k > lr_decay_threshold:
                learning_rate -= lr_decay_rate
            nmt.train(sess, train_dataset, dev_dataset, learning_rate)
        
        nmt.load(sess, nmt.save_path)
        
        
    if FLAGS.inference:
        infer_src_raw = load_data(FLAGS.data_path + 'src-infer.txt')
        infer_src = encode(src_vocab, infer_src_raw, True)
        infer_dataset = make_dataset('infer', infer_src)
        nmt.infer(sess, infer_dataset)
    else:
        str_acc, DFA_acc = nmt.eval(sess, test_dataset, True)
        print("test_str_equal_acc: ", str_acc, "test_DFA_euqal_acc: ", DFA_acc)

if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
