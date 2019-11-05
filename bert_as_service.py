import numpy as np

import bert_tokenization
from bert_serving.client import BertClient
from sentence_encoder import SentenceEncoder


BERT_BASE_DIR = 'external/bert/cased_L-24_H-1024_A-16/'
tokenizer = bert_tokenization.FullTokenizer(vocab_file=BERT_BASE_DIR+'vocab.txt',
                                            do_lower_case=False)


def bert_embed(bc, sents, merge_subtokens=True, merge_strategy='first'):
    sents_encodings_full = bc.encode(sents)
    sents_tokenized = [tokenizer.tokenize(s) for s in sents]

    sents_encodings = []
    for sent_tokens, sent_vecs in zip(sents_tokenized, sents_encodings_full):
        sent_encodings = []
        sent_vecs = sent_vecs[1:-1]  # ignoring [CLS] and [SEP]
        for token, vec in zip(sent_tokens, sent_vecs):
            layers_vecs = np.split(vec, 4)  # due to -pooling_layer -4 -3 -2 -1
            layers_sum = np.array(layers_vecs, dtype=np.float32).sum(axis=0)
            sent_encodings.append((token, layers_sum))
        sents_encodings.append(sent_encodings)

    if merge_subtokens:
        sents_encodings_merged = []
        for sent, sent_encodings in zip(sents, sents_encodings):

            sent_tokens_vecs = []
            for token in sent.split():  # these are preprocessed tokens

                token_vecs = []
                for subtoken in tokenizer.tokenize(token):
                    if len(sent_encodings) == 0:  # sent may be longer than max_seq_len
                        # print('ERROR: seq too long ?')
                        break

                    encoded_token, encoded_vec = sent_encodings.pop(0)
                    assert subtoken == encoded_token
                    token_vecs.append(encoded_vec)

                token_vec = np.zeros(1024)
                if len(token_vecs) == 0:
                    pass
                elif merge_strategy == 'first':
                    token_vec = np.array(token_vecs[0])
                elif merge_strategy == 'sum':
                    token_vec = np.array(token_vecs).sum(axis=0)
                elif merge_strategy == 'mean':
                    token_vec = np.array(token_vecs).mean(axis=0)

                sent_tokens_vecs.append((token, token_vec))

            sents_encodings_merged.append(sent_tokens_vecs)

        sents_encodings = sents_encodings_merged

    return sents_encodings


def bert_embed_sents(bc, sents, strategy='CLS_TOKEN'):
    sents_encodings_full = bc.encode(sents)
    sents_encodings = []
    for sent, sent_vec in zip(sents, sents_encodings_full):
        layers_vecs = np.split(sent_vec, 4)  # due to -pooling_layer -4 -3 -2 -1
        layers_sum = np.array(layers_vecs).sum(axis=0)
        sents_encodings.append((sent, layers_sum))

    return sents_encodings

class BertServiceSentenceEncoder(SentenceEncoder):
    def __init__(self, encoder_config):
        super(BertServiceSentenceEncoder, self).__init__(encoder_config)
        self.bc = BertClient()
        
    def token_embeddings(self, sents, return_ws_tokens=True):
        return bert_embed(self.bc, sents, merge_subtokens=return_ws_tokens,
                      merge_strategy=self.encoder_config.get('tok_merge_strategy', 'mean'))
