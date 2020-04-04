import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import nltk

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w
  
en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)
  
en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])

def max_length(tensor):
  return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# Try experimenting with the size of that dataset
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
# print(input_tensor_train[0])
# indices = np.arange(input_tensor_train.shape[0])
# np.random.shuffle(indices)

# a = input_tensor_train[indices]
# b = target_tensor_train[indices]
# alpha = 1.0
# lamb = np.random.beta(alpha, alpha)
# input_mix = lamb*input_tensor_train+(1-lamb)*a
# output_mix = lamb*target_tensor_train+(1-lamb)*b
# input_tensor_train = np.vstack((input_tensor_train,input_mix))
# target_tensor_train = np.vstack((target_tensor_train, output_mix))
# print(a[0])


# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))
      
print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape
# b, (i,o) = dataset.take(1)
# # print(b)
# print(i)

# # print(o)
class mix(tf.keras.layers.Layer):
  def __init__(self):
    super(mix, self).__init__()
    # self.lamb = lamb
  def call(self, x, lamb, index ,is_train):
    if not is_train:
      return x
    # indices = tf.range(start =0, limit = 64)#tf.shape(x)[0])
    # shuffled_indices = tf.random.shuffle(indices)
    # print(indices)
    a = tf.gather(x, index)
    out = tf.math.add(tf.math.scalar_mul(lamb,x),tf.scalar_mul((1.0-lamb),a))
    return out


class mix_up_embed(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(mix_up_embed, self).__init__(name='')
    self.e=self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.m = mix()
  def call(self,input_tensor, lamb, index, is_train=True):
      y = self.e(input_tensor)
      y = self.m(y,lamb, index, is_train)
      return y
    
  


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = mix_up_embed(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden, lamb, index, is_train=True):
    x = self.embedding(x, lamb,index,is_train)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
    
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
indices = tf.range(start =0, limit = 64)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden, np.random.beta(1,1), index=indices)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = mix_up_embed(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output, lamb, index,is_train=True):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x,lamb,index, is_train)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
    
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
indices = tf.range(start =0, limit = 64)
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output, np.random.beta(1,1), index=indices)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
  
checkpoint_dir = './training_checkpoints'#'drive/My Drive/nmt_checkpoint/''./training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
                                 
@tf.function
def train_step(inp, targ, enc_hidden, lamb, indices):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden, lamb, index=indices)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, lamb,index=indices)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))
#   print('a')

  return batch_loss

def evaluate(inputs):
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  attention_plot = np.zeros((max_length_targ, max_length_inp))

#   sentence = preprocess_sentence(sentence)
#   print("sentence=>",sentence)
  

  
#   inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
#   print(inputs)
#   inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                          maxlen=max_length_inp,
#                                                          padding='post')
  inputs = np.reshape(inputs,(1,-1))
  inputs = tf.convert_to_tensor(inputs)
#   print("shape",inputs.shape)
  result = '<start> '

  hidden = [tf.zeros((1, units))]
  indices = tf.range(start =0, limit = 64)
  enc_out, enc_hidden = encoder(inputs, hidden, 1.0, indices, False)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out,1.0, indices, False)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()
    result += targ_lang.index_word[predicted_id] + ' '
    if targ_lang.index_word[predicted_id] == '<end>':
      return result, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, attention_plot

def bleu():
#   targ_lang, inp_lang = create_dataset(path, num_examples)
#   print(targ_lang[-1])
  targ_lang_list = []
  hyp_targ_list = []
  for sentence in target_tensor_val[:100]:
    x = ''
    for word in sentence:
    #   print(word)
      x = x+targ_lang.index_word[word]+' '
      if targ_lang.index_word[word] == '<end>':
        break
    # print(x)
    hyp_targ_list.append(x)
#   print(hyp_targ_list)
#   hyp_lang_list = [sent.split(' ') for sent in targ_lang]
  for sentence in input_tensor_val[:100]:
    # print("test",sentence.shape)  
    result, attention_plot = evaluate(sentence)
    targ_lang_list.append(result)
#   print(targ_lang_list)
  hyp_list = [sentence.split(' ') for sentence in hyp_targ_list]
  pred_list = [sentence.split(' ') for sentence in targ_lang_list]
#   print(hyp_list[0])
#   print(pred_list[0])
#   print(hyp_list[1])
#   print(pred_list[1])
#   print(hyp_list[2])
#   print(pred_list[2])
  k = nltk.translate.bleu_score.sentence_bleu([hyp_list[0]], pred_list[0])
  print('k',k)
  blue_list = []
  for i in range(100):
    # print(hyp_list[i])
    # print(pred_list[i])
    blue_list.append(nltk.translate.bleu_score.sentence_bleu([hyp_list[i]], pred_list[i]))
  print('val', np.mean(blue_list))
  return np.mean(blue_list)

EPOCHS = 50
li = []
li2=[]
for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    # alpha = 1.0
    # indices = tf.range(start =0, limit = tf.shape(inp).numpy()[0])
    # shuffled_indices = tf.random.shuffle(indices)
    # # print(indices)
    # a = tf.gather(inp, shuffled_indices)
    # b = tf.gather(targ, shuffled_indices)
    lamb = np.random.beta(1,1)
    indices = tf.range(start =0, limit = 64)#tf.shape(x)[0])
    shuffled_indices = tf.random.shuffle(indices)
    lamb = tf.convert_to_tensor(lamb, dtype=tf.float32)
    # alpha = 1.0
    # lamb = np.random.beta(alpha, alpha)
    # input_mix = lamb*inp+(1-lamb)*a
    # output_mix = lamb*targ+(1-lamb)*b
    # inp = input_mix
    # targ = output_mix
    # input_tensor_train = np.vstack((input_tensor_train,input_mix))
    # target_tensor_train = np.vstack((target_tensor_train, output_mix))
    # print(a[0])


    batch_loss = train_step(inp, targ, enc_hidden, lamb, indices)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
      
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  li.append(batch_loss.numpy())
  li2.append(bleu())
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

from matplotlib import pyplot as plt 
x = np.arange(1,51)
y = li
plt.subplot(2,1,1)
plt.plot(np.array(x),np.array(y))
plt.subplot(2,1,2)
plt.plot(np.array(x), np.array(li2))

def evaluate_old(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)
  print("shape", inputs.shape)
  result = ''

  hidden = [tf.zeros((1, units))]
  indices = tf.range(start =0, limit = 64)
  enc_out, enc_hidden = encoder(inputs, hidden, 1.0, indices, False)
  print(enc_out.shape)
  print(enc_hidden.shape)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out,1.0, indices,False)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
  
  def translate(sentence):
  result, sentence, attention_plot = evaluate_old(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate(u'hace mucho frio aqui.')
