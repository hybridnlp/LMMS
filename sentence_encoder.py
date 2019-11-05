import torch
import logging
from transformers import *
import numpy as np
from contextlib import contextmanager
import copy

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

lmms_default_encoder_cfg = {
    'model_name_or_path': 'bert-large-cased',
    'model_arch': 'BERT',
    'do_lower_case': False,
    'min_seq_len': 0,
    'max_seq_len': 512,
    'pooling_strategy': 'NONE',
    'pooling_layer': [-4, -3, -2, -1],
    'tok_merge_strategy': 'mean'
}


def infer_arch_from_name(model_name):
    """Given a transfromers model name, guess the model architecture.
    For example `infer_arch_from_nmae('bert-large-cased) # BERT`
    :param model_name: a model name supported by huggingface transformers
    :returns: an architecture name like 'BERT', 'RoBERTa', etc.
    :rtype: str
    """
    if model_name.startswith('bert-'):
        return "BERT"
    elif model_name.startswith('roberta-'):
        return "RoBERTa"
    elif model_name.startswith('xlnet-'):
        return "XLNet"
    elif model_name.startswith('gpt2-'):
        return "GPT2"
    elif model_name.startswith('ctrl'):
        return "CTRL"
    else:
        raise ValueError(model_name)


def tokenize(encoder, text):
    def needs_begin_whitespace():
        return encoder.encoder_config['model_arch'] in ['RoBERTa', 'GPT2', 'CTRL']

    if needs_begin_whitespace():
        text = ' ' + text
    return encoder.tokenizer.tokenize(text)

def tokenize_tail_wstok(encoder, wstok):
    """some tokenizers (eg GPT2) encode tokens differently depending on whether they are the 
    first token in a sequence or not. This method should be called for non-first tokens
    """
    if encoder.encoder_config['model_arch'] in ['GPT2']:
        tokens = tokenize(encoder, 'my %s' % wstok)
        return tokens[1:] # ismply remove first subtoken (my is tokenized as 'my')
    else:
        return tokenize(encoder, wstok)

### basic encoding
def pad_encode(encoder, text, max_length):
  """creates token ids of a uniform sequence length for a given sentence"""
  tokenizer = encoder.tokenizer
  has_spec_toks = True if not hasattr(encoder, 'sent_special_tokens') else len(encoder.sent_special_tokens) > 0
  # TODO can we use tokenizer.encode(text, add_special_tokens=True, max_length=max_length) ?
  
  tok_ids = tokenizer.convert_tokens_to_ids(tokenize(encoder, text))
  #tok_ids2 = tokenizer.add_special_tokens_single_sentence(tok_ids) # pyt_transformers
  tok_ids2 = tokenizer.build_inputs_with_special_tokens(tok_ids) if has_spec_toks else tok_ids # transformers
  att_mask = [1 for _ in range(len(tok_ids2))]
  n_spectoks = len(tok_ids2) - len(tok_ids)
  if len(tok_ids2) > max_length: # need to truncate
    #print('Truncating from', len(tok_ids2))
    n_to_trunc = len(tok_ids2) - max_length
    #tok_ids2 = tokenizer.add_special_tokens_single_sentence(tok_ids[:-n_to_trunc])
    tok_ids2 = tokenizer.build_inputs_with_special_tokens(tok_ids[:-n_to_trunc]) if has_spec_toks else tok_ids[:-n_to_trunc]
    att_mask = [1 for _ in range(len(tok_ids2))]
  elif len(tok_ids2) < max_length: # need to pad
    padding = []
    pad_tok_id = 0 if tokenizer._pad_token is None else tokenizer.pad_token_id
    for i in range(len(tok_ids2), max_length):
      # some models don't have a defined `pad_token_id`, so use 0 by default 
      # this will be masked anyway, so shouldn't really matter
      padding.append(pad_tok_id) 
    att_mask = att_mask + [0 for _ in range(len(padding))]
    tok_ids2 = tok_ids2 + padding
  assert len(tok_ids2) == max_length
  assert len(att_mask) == max_length
  return tok_ids2, att_mask


def calc_max_len(encoder, sents):
  tokenizer = encoder.tokenizer
  maxlen = max([len(tokenize(encoder, text)) for text in sents])
  if maxlen is None:
    maxlen = encoder.encoder_config.get('max_seq_len', 512)
  else:
    maxlen = min(maxlen, encoder.encoder_config.get('max_seq_len', 512))


def tokenize_batch(encoder, sents):
  """Produces a tensor of padded token ids for a given list of sentences
  Sentences may need to be padded or truncated, depending on the `max_seq_len`
  of the `encoder`.
  """
  assert type(sents) == list
  maxlen = calc_max_len(encoder, sents)

  padded_tok_ids = [pad_encode(encoder, s, maxlen)[0] for s in sents]
  att_masks = [pad_encode(encoder, s)[1] for s in sents]
  input_ids = torch.tensor(padded_tok_ids)
  att_masks = torch.tensor(att_masks)
  #logging.info("Input batch %s " % str(input_ids.shape))
  cfg = encoder.encoder_config
  if cfg.get('debug', False): print(input_ids.shape)
  if torch.cuda.is_available() and not cfg.get('cpu', False):
    input_ids = input_ids.cuda()
    att_masks = att_masks.cuda()
  return input_ids, att_masks

def embedding_from_bert_output(encoder, bert_output, att_masks=None):
  """Given the output tensor from the encoder's model, return embeddings.
  This is done based on the `encoder`'s parameters `pooling_strategy` and 
  `pooling_layer`. This is implemented based on 
  see also https://github.com/hanxiao/bert-as-service/blob/master/server/bert_serving/server/graph.py
  """
  cfg = encoder.encoder_config

  assert len(bert_output) == 3, "Expecting 3 outputs, make sure model outputs hidden states"
  last_layer, pooled, all_encoder_layers = bert_output
  
  pooling_layer = cfg.get('pooling_layer', [-2]) 
  if len(pooling_layer) == 1:
    encoder_layer = all_encoder_layers[pooling_layer[0]]
  else: # if multiple layers requested, concatenate them
    all_layers = [all_encoder_layers[layer_idx] for layer_idx in pooling_layer]
    encoder_layer = torch.cat(all_layers, -1)

  strategy = encoder.encoder_config.get('pooling_strategy', 'REDUCE_MEAN')
  if strategy == "REDUCE_MEAN":
    # TODO: bert-as-service also uses a mask tensor... take into account?
    # ie use the input_mask param
    pooled_layer = torch.sum(encoder_layer, dim=1) / (encoder_layer.shape[1] + 1e-10)
    if debug: print('pooled layers %s of %s' % (pooling_layer, len(hidden_layers)), 
                    pooled_layer.shape,
                    'pooled from', encoder_layer.shape)
    return pooled_layer
  elif strategy == "NONE":
    return encoder_layer
  else:
    raise NotImplementedError("Strategy: " + strategy)

def find_out_tok_dim(encoder):
  """Figures out the dimension of a single subtoken embedding"""
  input_ids, att_masks = tokenize_batch(encoder, ["Test sentence to encode"])
    
  model = encoder.model
  model.eval() # needed to deactivate any Dropout layers

  #logging.info("Figuring out the output subtoken dimension for %s" % str(model))
  with torch.no_grad():
    model_out = model(input_ids, attention_mask=att_masks)
  final_layer = model_out[0]
  batch_size, seqlen, dim = final_layer.shape
  return dim

def find_sent_special_tokens(encoder):
  """Figure out the special tokens added by the tokenizer to a single sentence"""
  tokenizer = encoder.tokenizer
  tok_ids = tokenizer.convert_tokens_to_ids(tokenize(encoder, "Test text"))
  #tok_ids2 = tokenizer.add_special_tokens_single_sentence(tok_ids)
  tok_ids2 = tokenizer.build_inputs_with_special_tokens(tok_ids)
  if len(tok_ids) == len(tok_ids2):
    return []
  
  result = []
  orig_idx = 0
  for idx, tid2 in enumerate(tok_ids2):
    if orig_idx < len(tok_ids) and tid2 == tok_ids[orig_idx]: 
      # not a special token, advance
      orig_idx += 1
    else: #special token
      result.append(
        {
           'index': idx if orig_idx == 0 else idx - len(tok_ids2),
           'tok_id': tid2,
           'tok': tokenizer.convert_ids_to_tokens([tid2])[0] })
  return result

def simple_batch_encode(encoder, sents):
  """Uses the encoder to tokenize and pass a batch of sentences through a 
  transformer model. Returns a numpy array.
  :returns numpy array of dim either:
    `(len(sents), max_seq_len, emb_dim)` when `pooling_strategy` is `NONE` OR
    `(len(sents), emb_dim)` when `pooling_strategy` was something else
  """
  input_ids, att_masks = tokenize_batch(encoder, sents)

  model = encoder.model
  model.eval() # needed to deactivate any Dropout layers

  with torch.no_grad():
    model_out = model(input_ids, attention_mask=att_masks)
  
  return embedding_from_bert_output(encoder, model_out, att_masks), att_masks

def encode(encoder, sents):
  """Returns an array of encoded sentences using the `encoder`
  For token-level encodings, the input `sents` are tokenized and padded or 
  truncated to a fixed length (specified in the `encoder.encoder_config`)
  :param encoder an instance of `SentenceEncoder`
  :param sents a list of sentences (`str` instances)
  :returns a numpy array of dimensions `(len(sents), max_len, emb_dim)` 
  """
  batch_out, att_masks = simple_batch_encode(encoder, sents)
  return batch_out.cpu().numpy(), att_masks.cpu().numpy()


#### Token-level embeddings
def read_sub_token_embs(sent_encoder, ws_token, sent_subtok_embs, start_idx):
  """Reads the sub-token embeddings corresponding to a whitespace token from a
  `sent_subtok_embs`, which should start from index `start_idx`.
  """
  subtok_vecs = []
  expected_subtoks = tokenize(sent_encoder, ws_token) if start_idx == 0 else tokenize_tail_wstok(sent_encoder, ws_token)
  end_idx = start_idx + len(expected_subtoks)
  for subtoken, (encoded_token, encoded_vec) in zip(
      expected_subtoks, sent_subtok_embs[start_idx:end_idx]):
    assert subtoken == encoded_token, '%s != %s, %s -> (%s) != %s' % (
        subtoken, encoded_token, ws_token, str(expected_subtoks), str(sent_subtok_embs[start_idx:end_idx]))
    subtok_vecs.append(encoded_vec)
  return subtok_vecs, end_idx

def merge_subtoken_vecs(sent_encoder, subtoken_vecs):
  """ Merges a list of subtoken_vecs into a single embedding.
  It is assumed that `subtoken_vecs` are the subtoken vectors corresponding to a
  whitespace token.
  """
  merge_strategy = sent_encoder.encoder_config.get('tok_merge_strategy', 'mean')
  if len(subtoken_vecs) == 0:
    return np.zeros(sent_encoder.subtoken_dim()) 
  elif merge_strategy == 'first':
    return np.array(subtoken_vecs[0])
  elif merge_strategy == 'sum':
    return np.array(subtoken_vecs).sum(axis=0)
  elif merge_strategy == 'mean':
    return np.array(subtoken_vecs).mean(axis=0)
  return token_vec

def subtok_to_wstok_vecs(sent_encoder, sent, sent_subtok_embs):
  """ Merges subtoken encodings for a sentence into whitespace token encodings 
  """
  sent_tokens_vecs = []
  sent_subtok_idx = 0  # keep track of where we are in the `sent_encodings` while iterating tokens
  for token in sent.split():  # these are preprocessed tokens
    subtoken_vecs, sent_subtok_idx = read_sub_token_embs(
        sent_encoder, token, sent_subtok_embs, sent_subtok_idx)
    token_vec = merge_subtoken_vecs(sent_encoder, subtoken_vecs)
    sent_tokens_vecs.append((token, token_vec))
  return sent_tokens_vecs

def merge_subtokens(encoder, sents, sents_subtok_embs):
  """ Converts subtoken embeddings into whitespace token embeddings for a batch 
  of sentences
  """
  merge = lambda sent, sent_subtok_vecs: subtok_to_wstok_vecs(
      encoder, sent, sent_subtok_vecs)
  return [merge(sent, sent_subtok_embs) for sent, sent_subtok_embs in zip(sents, sents_subtok_embs)]


def remove_special_tokens(encoder, sent_full_vecs):
  begin, end = -1, len(sent_full_vecs)
  for spec_tok in encoder.sent_special_tokens:
    idx = spec_tok['index']
    if idx >= 0: 
        begin = max(begin, idx)
    elif idx < 0:
        end = min(end, idx)
  #logging.info("Removing special characters %d:%d" % (begin+1, end))
  return sent_full_vecs[begin+1:end]


def subtok_embed_sentence(encoder, sent_subtokens, sent_full_vecs, sent_att_mask):
  """Converts full vectors for a sentence into a list of subtoken vectors
  :param sent_full_vecs numpy array of shape (seqlen, dim)
  :param sent_att_mask numpy array of shape (seqlen, ) indicates which subtokens are padding"""
  sent_subtok_vecs = []
  #logging.info("Filtering full vecs %s with %s" % (sent_full_vecs.shape, sent_att_mask.shape))
  sent_wspec_char_vecs = sent_full_vecs[sent_att_mask == 1,:]
  #logging.info("Filtered vecs %s" % str(sent_wspec_char_vecs.shape))
  sent_vecs = remove_special_tokens(encoder, sent_wspec_char_vecs) # [1:-1]  # remove special tokens (eg [CLS] and [SEP] in BERT)
  assert len(sent_vecs) == len(sent_subtokens), "expecting %s == %s" % (
    len(sent_vecs), len(sent_subtokens))
  pooling_layer = encoder.encoder_config.get('pooling_layer', [-2]) 
  for subtok, vec in zip(sent_subtokens, sent_vecs):
    if len(pooling_layer) == 1:
      sent_subtok_vecs.append((subtok, vec))
    else: # multiple layers concatenated, so 
      #print("Splitting", vec.shape, 'into', len(pooling_layer))
      layers_vecs = np.split(vec, len(pooling_layer))  # e.g -pooling_layer -4 -3 -2 -1
      layers_sum = np.array(layers_vecs, dtype=np.float32).sum(axis=0)
      sent_subtok_vecs.append((subtok, layers_sum))
  return sent_subtok_vecs

def subtok_embed(encoder, sents):
  """Encode a batch of sentences at the subtoken level. Similar to `encode` but:
   * without special tokens
   * if `pooling_layer` specifies multiple layers, the embeddings are summed so 
    that the embedding dimension is the same as that of a single token.
   * instead of returning a numpy array, this returns a list of sub-tokenized 
    sentence embeddings. Where each sentence is a list of tuples `(subtoken, emb)` 
  """
  sents_encodings = []
  batch_full_embs, batch_att_masks = encode(encoder, sents)
  for sent_toks, sent_vecs, sent_att_mask in zip(
      [tokenize(encoder, s) for s in sents],
      batch_full_embs, batch_att_masks):
    sents_encodings.append(subtok_embed_sentence(encoder, sent_toks, sent_vecs, sent_att_mask))
  return sents_encodings

def token_embed(sent_encoder, sents, return_ws_tokens=True):
  """Returns a list of tokenized sentences with embeddings for each token
  :param sent_encoder a `SentenceEncoder` instance
  :param sents a list of **preprocessed** sentences so that sentence.split() 
    returns individual tokens
  """
  sents_subtok_vecs = subtok_embed(sent_encoder, sents)
  if return_ws_tokens:
    return merge_subtokens(sent_encoder, sents, sents_subtok_vecs)
  else:
    return sents_subtok_vecs



#### Main class
class SentenceEncoder():
    def __init__(self, encoder_config):
        self.encoder_config = copy.deepcopy(encoder_config)
        
    def token_embedding(self, sents, return_ws_tokens=True):
        raise NotImplementedError()
        
    def num_special_tokens(self):
        # BERT uses 2 special tokens [CLS] and [SEP]
        return 2 # other implementation may have a different number
    
    def subtoken_dim(self):
        """Returns the size of the subtoken embeddings e.g. bert-large returns 1024"""
        raise NotImplementedError()
    
    def is_valid_len(self, sentence):
      bert_tokens = tokenize(self, sentence)
      special_toks = self.num_special_tokens()
      # bert-as-service default is 25, but LMMS assumes 512
      max_len = self.encoder_config.get('max_seq_len', 512)  
      min_len = self.encoder_config.get('min_seq_len', 3)
      bl = len(bert_tokens)
      #logging.info("Sentence len %d min %d max %d" % (bl, min_len, max_len))
      return bl <= (max_len - special_toks) and bl > (min_len - special_toks)  

    
@contextmanager
def min_maxlen_encoder(sent_encoder, minlen, maxlen):
    """Context manager function to temporarily modify the `min_seq_len` and `max_seq_len`
    of a SentenceEncoder."""
    cfg = sent_encoder.encoder_config
    orig_min = cfg.get('min_seq_len', 3)
    orig_max = cfg.get('max_seq_len', 512)
    try:
        cfg['min_seq_len'] = minlen
        cfg['max_seq_len'] = maxlen
        yield sent_encoder
    finally:
        cfg['min_seq_len'] = orig_min
        cfg['max_seq_len'] = orig_max
        
        
def create_tx_model(model_name, encoder_config):
    if model_name in ['bert-base-cased', 'bert-large-cased', 
                      'bert-base-multilingual-cased', 'bert-base-chinese',
                     'bert-base-german-cased',
                     'bert-large-cased-whole-word-masking',
                     'bert-large-cased-whole-word-masking-finetuned-squad',
                     'bert-base-cased-finetuned-mrpc']:
        return BertModel.from_pretrained(model_name, output_hidden_states=True)
    elif model_name in ['bert-base-uncased', 'bert-large-uncased', 
                        'bert-base-multilingual-uncased',
                       'bert-large-uncased-whole-word-masking',
                       'bert-large-uncased-whole-word-masking-finetuned-squad']:
        return BertModel.from_pretrained(model_name, output_hidden_states=True)
    elif model_name in ['roberta-base', 'roberta-large', 'roberta-large-mnli']:
        return RobertaModel.from_pretrained(model_name, output_hidden_states=True)
    elif model_name in ['xlnet-large-cased', 'xlnet-base-cased']:
        return XLNetModel.from_pretrained(model_name, output_hidden_states=True)
    elif model_name in ['gpt2-medium', 'gpt2-large']:
        return GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    elif model_name in ['ctrl']:
        return CTRLModel.from_pretrained(model_name, output_hidden_states=True)
    else:
        raise ValueError(model_name)
        
def create_tokenizer(model_name, encoder_config):
    if model_name in ['bert-base-cased', 'bert-large-cased', 
                      'bert-base-multilingual-cased', 'bert-base-chinese',
                     'bert-base-german-cased',
                     'bert-large-cased-whole-word-masking',
                     'bert-large-cased-whole-word-masking-finetuned-squad',
                     'bert-base-cased-finetuned-mrpc']:
        return BertTokenizer.from_pretrained(
            model_name, do_lower_case=encoder_config.get('do_lower_case', False))
    elif model_name in ['bert-base-uncased', 'bert-large-uncased', 
                        'bert-base-multilingual-uncased',
                       'bert-large-uncased-whole-word-masking',
                       'bert-large-uncased-whole-word-masking-finetuned-squad']:
        return BertTokenizer.from_pretrained(
            model_name, do_lower_case=encoder_config.get('do_lower_case', True))
    elif model_name in ['roberta-base', 'roberta-large', 'roberta-large-mnli']:
        return RobertaTokenizer.from_pretrained(
            model_name, do_lower_case=encoder_config.get('do_lower_case', False))
    elif model_name in ['xlnet-large-cased', 'xlnet-base-cased']:
        return XLNetTokenizer.from_pretrained(
            model_name, do_lower_case=encoder_config.get('do_lower_case', False))
    elif model_name in ['gpt2-medium', 'gpt2-large']:
        return GPT2Tokenizer.from_pretrained(model_name)
    elif model_name in ['ctrl']:
        return CTRLTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(model_name)
        
class TransformerSentenceEncoder(SentenceEncoder):
  """Class that encapsulates a transformer tokenizer and model and knows how
    to generate sentence (and token-level) encodings based on those.
  """

  def __init__(self, encoder_config):
    """
    :param encoder_config dict with keys
      `model_arch` str with one of the supported transformer models e.g. BERT
      `model_name_or_path`: pre-trained model name supported by pytorch 
        transformer models, or path to folder where a pre-trained model was saved
      `do_lower_case`: whether to convert sequences to lowercase during tokenization,
        by default False, but if you are using a uncased model, you should set 
        this to True.
      `max_seq_len` int length of the batches to send to the model. Longer 
        sequences will be truncated, shorter will be padded.
      `pooling_strategy` str sentence-level pooling strategy, see bert-as-service 
        for the available options. For token-level embeddings, use `NONE`.
      `pooling_layer` seq of int indices of the transformer encoder layers to 
        return. When multiple layers specified, by default we concatenate the 
        embeddings during basic encoding, but we merge them for subtoken and 
        token level embeddings.
      `tok_merge_strategy` str how to combine embeddings for subtokens into a 
        whitespace token embedding. 
    """
    super(TransformerSentenceEncoder, self).__init__(encoder_config)
    logging.info("Creating TransformerSentenceEncoder")
    model_name = self.encoder_config['model_name_or_path']
    self.tokenizer = create_tokenizer(model_name, self.encoder_config)
    # TODO: optionally optimize the model? i.e. specify max_len at the model level?
    self.model = create_tx_model(model_name, self.encoder_config) 
    
    for param in self.model.parameters(): #freeze all params (no need to train this)
        param.requires_grad = False
    if torch.cuda.is_available() and not self.encoder_config.get('cpu', False):
        self.model = self.model.cuda()
    
    # figure out dimensions and sentence special tokens for this model/tokenizer
    self.tok_dim = find_out_tok_dim(self)
    self.sent_special_tokens = find_sent_special_tokens(self)
    self.encoder_config['sent_special_tokens'] = self.sent_special_tokens
    self.encoder_config['tok_dim'] = self.tok_dim
    logging.info(
        "Created TransformerSentenceEncoder\n\tconfig %s\n\ttok_dim:%d\n\tsent_special_tokens%s" % (
        str(self.encoder_config), self.tok_dim, str(self.sent_special_tokens)))

  def token_embeddings(self, sents, return_ws_tokens=True):
    return token_embed(self, sents, return_ws_tokens=return_ws_tokens)

  def num_special_tokens(self):
    # BERT uses 2 special tokens [CLS] and [SEP]
    return len(self.sent_special_tokens) # other implementation may have a different number
    
  def subtoken_dim(self):
    """Returns the size of the subtoken embeddings e.g. bert-large returns 1024"""
    return self.tok_dim
