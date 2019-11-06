import os
import logging
import argparse
from time import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
from nltk.corpus import wordnet as wn

from sentence_encoder import TransformerSentenceEncoder, infer_arch_from_name
import mwtok
from vectorspace import SensesVSM
from vectorspace import get_sk_pos
from train import get_sense_mapping

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def read_xml_sentence(sent_idx, xml_sentence, sense_mapping):
    inst = {f: [] for f in ['tokens', 'tokens_mw', 'lemmas', 'senses', 'pos', 'gold_sensekeys']}
    for e in xml_sentence:
        inst['tokens_mw'].append(e.text)
        #if e.get('lemma') is None:
        #    raise ValueError("No lemma for %s, %s" % (list(e.keys()), e.text))
        inst['lemmas'].append(e.get('lemma'))
        sense = e.get('id')
        inst['senses'].append(sense)
        inst['gold_sensekeys'].append(sense_mapping.get(sense, None))
        inst['pos'].append(e.get('pos'))

    inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

    # handling multi-word expressions, mapping allows matching tokens with mw features
    idx_map_abs = []
    idx_map_rel = [(i, list(range(len(t.split()))))
                   for i, t in enumerate(inst['tokens_mw'])]
    token_counter = 0
    for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
        idx_tokens = [i+token_counter for i in idx_tokens]
        token_counter += len(idx_tokens)
        idx_map_abs.append([idx_group, idx_tokens])

    inst['tokenized_sentence'] = ' '.join(inst['tokens'])
    inst['idx_map_abs'] = mwtok.calc_idx_map_abs(inst['tokens_mw'])
    inst['idx'] = sent_idx
    return inst


def read_xml_sents(xml_path, as_entry_fn=lambda sent_idx, sent_et: sent_et):
    for elts in ET.parse(xml_path).getroot():
        for sent_idx, sent_elt in enumerate(elts):
            yield as_entry_fn(sent_idx, sent_elt)


def wnet_wsd_fw_set_generator(wsd_fw_set_path, sense_mapping):
    """Parse XML of split set and return generator of instances (dict).
    :returns a list of evaluation sentences, each is a dict with keys
        `tokens`, `tokens_mw`, `lemmas`, `senses`, `pos` and 
        `tokenized_sentence`, `idx_map_abs` and `idx`.
    """
    return read_xml_sents(wsd_fw_set_path, 
                          lambda idx, sent: read_xml_sentence(idx, sent, sense_mapping))


def wsd_fw_set_generator(wsd_fw_set_path, sense_mapping=None):
    """Parse XML of split set and return generator of instances (dict).
    :returns a list of evaluation sentences, each is a dict with keys
        `tokens`, `tokens_mw`, `lemmas`, `senses`, `pos` and 
        `tokenized_sentence`, `idx_map_abs` and `idx`.
    """
    assert wsd_fw_set_path.endswith('.data.xml')
    return wnet_wsd_fw_set_generator(wsd_fw_set_path, sense_mapping)
    
@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


def get_id2sks(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2sks = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
    """Runs the official java-based scorer of the WSD Evaluation Framework."""
    cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
                                          '%s/%s.gold.key.txt' % (test_set, test_set),
                                          '../../../../' + results_path)
    print(cmd)
    os.system(cmd)


def chunks(gen, n):
    """Yield successive n-sized chunks from given generator."""
    batch = []
    for it in gen:
        batch.append(it)
        if len(batch) == n:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def str_scores(scores, n=3, r=5):
    """Convert scores list to a more readable string."""
    return str([(l, round(s, r)) for l, s in scores[:n]])


def encoder_config(args):
    return {
        'model_name_or_path': args.pytorch_model,
        'model_arch': infer_arch_from_name(args.pytorch_model),
        'min_seq_len': args.min_seq_len,
        'max_seq_len': args.max_seq_len,
        'pooling_strategy': 'NONE',
        'pooling_layer': args.pooling_layer, #[-4, -3, -2, -1],
        'tok_merge_strategy': args.merge_strategy}


def build_encoder(args):
    backend = args.backend
    enc_cfg = encoder_config(args)
    if backend == 'bert-as-service':
        from bert_as_service import BertServiceSentenceEncoder
        return BertServiceSentenceEncoder(enc_cfg)
    elif backend == 'transformers':
        return TransformerSentenceEncoder(enc_cfg)
    else:
        raise NotImplementedError("backend " + backend)


def load_fast_text(args):
    logging.info('SensesVSM requires fastText')
    if args.ft_path != '':
        logging.info('Loading pretrained fastText ...')
        import fastText  # importing here so that fastText is an optional requirement
        result = fastText.load_model(args.ft_path)
        logging.info('Loaded pretrained fastText')
        return result
    else:
        logging.critical('fastText model is undefined and expected by SensesVSM.')
        raise Exception('Input Failure')


def testset_generator(args):
    """
    Load evaluation instances and gold labels.
    Gold labels (sensekeys) only used for reporting accuracy during evaluation.
    """
    wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (args.test_set, args.test_set)
    wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (args.test_set, args.test_set)
    id2senses = get_sense_mapping(wsd_fw_gold_path)
    return wsd_fw_set_generator(wsd_fw_set_path, id2senses)


def empty_eval_state():
    return {
        'n_instances': 0,
        'n_correct': 0,
        'n_unk_lemmas': 0,
        'correct_idxs': [],
        'num_options': [],
        'failed_by_pos': defaultdict(list),
        'pos_confusion': empty_pos_confusion()
    }


def merge_eval_state(s, s2):
    #logging.info("Before\n\t%s\n\t%s" % (str(s), str(s2)))
    for f in ['n_instances', 'n_correct', 'n_unk_lemmas']:
        s[f] += s2.get(f, 0)
    for f in ['correct_idxs', 'num_options']:
        s[f] += s2.get(f, [])
    poss = ['NOUN', 'VERB', 'ADJ', 'ADV']
    for from_pos in poss:
        for to_pos in poss:
            s['pos_confusion'][from_pos][to_pos] += s2['pos_confusion'][from_pos][to_pos]
    #logging.info("After\n\t%s" % str(s))
    return s


def empty_pos_confusion():
    pos_confusion = {}
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}
    return pos_confusion


def build_mwtok_emb(sent_bert, tok_idxs, 
                    lemma, tokens, 
                    ft_model, 
                    senses_vsm, args):
    vector = np.array([sent_bert[i][1] for i in tok_idxs]).mean(axis=0)
    vector = vector / np.linalg.norm(vector)
                    
    """
    Fetch (or compose) static embedding if it's expected.
    Uses lemma by default unless specified otherwise by CLI parameter.
    This lemma is a gold feature of the evaluation datasets.
    """
    static_vector = None
    if ft_model is not None:
        if args.use_lemma:
            static_vector = ft_model.get_word_vector(lemma)
        else:
            static_vector = ft_model.get_word_vector('_'.join(tokens))
        static_vector = static_vector / np.linalg.norm(static_vector)

    """
    Compose test-time embedding for matching with sense embeddings in SensesVSM.
    Test-time embedding corresponds to stack of contextual and (possibly) static embeddings.
    Stacking composition performed according to dimensionality of sense embeddings.
    """
    if senses_vsm.ndims == 1024:
        vector = vector

    # duplicating contextual feature for cos similarity against features from
    # sense annotations and glosses that belong to the same NLM
    elif senses_vsm.ndims == 1024+1024:
        vector = np.hstack((vector, vector))

    elif senses_vsm.ndims == 300+1024 and static_vector is not None:
        vector = np.hstack((static_vector, vector))

    elif senses_vsm.ndims == 300+1024+1024 and static_vector is not None:
        vector = np.hstack((static_vector, vector, vector))

    return vector / np.linalg.norm(vector)


def find_matches(senses_vsm, vector, lemma, postag, args):
    """
    Matches test-time embedding against sense embeddings in SensesVSM.
    use_lemma and use_pos flags condition filtering of candidate senses.
    Matching is actually cosine similarity (most similar), or 1-NN.
    """
    if args.use_lemma and lemma not in senses_vsm.known_lemmas:
        return []
    elif args.use_lemma and args.use_pos:  # the usual for WSD
        return senses_vsm.match_senses(vector, lemma, postag, topn=None)
    elif args.use_lemma:
        return senses_vsm.match_senses(vector, lemma,   None, topn=None)
    elif args.use_pos:
        return senses_vsm.match_senses(vector,  None, postag, topn=None)
    else:  # corresponds to Uninformed Sense Matching (USM)
        return senses_vsm.match_senses(vector,  None,   None, topn=None)

    
def eval_mwtok(sent_info, mw_idx, tok_idxs, sent_bert, 
               senses_vsm, args, ft_model):
    """Evaluates a single multiword token prediction from sent_bert against the ground truth 
    in sent_info agains the ground truth.
    :returns triple (eval_status, sense, prediction)
    """
    result = empty_eval_state()
    curr_sense = sent_info['senses'][mw_idx]

    if curr_sense is None:
        return result, curr_sense, None

    curr_lemma = sent_info['lemmas'][mw_idx]

    if args.use_lemma and curr_lemma not in senses_vsm.known_lemmas:
        logging.info("Lemma %s not in vocab of %s" % (curr_lemma, len(senses_vsm.known_lemmas)))
        result['n_unk_lemmas'] = 1
        return result, curr_sense, None  # skips hurt performance in official scorer

    #logging.info("Found lemma %s and sense %s" % (curr_lemma, curr_sense))
    
    curr_postag = sent_info['pos'][mw_idx]
    curr_tokens = [sent_info['tokens'][i] for i in tok_idxs] # ws tokens
    
    curr_vector = build_mwtok_emb(sent_bert, tok_idxs, 
                    curr_lemma, curr_tokens, 
                    ft_model, senses_vsm, args)

    #debugs_f.write('%s %s %s %s\n' % (
    #    curr_sense, curr_lemma, curr_vector[0], curr_vector[1]))
    
    matches = find_matches(senses_vsm, curr_vector, curr_lemma, curr_postag, args)

    result['num_options'].append(len(matches)) # num_options.append(len(matches))

    # predictions can be further filtered by similarity threshold or number of accepted neighbors
    # if specified in CLI parameters
    preds = [sk for sk, sim in matches if sim > args.thresh][:args.k]

    """
    Processing additional performance metrics.
    """
    # check if our prediction(s) was correct, register POS of mistakes
    result['n_instances'] = 1  # n_instances += 1
    wsd_correct = False
    gold_sensekeys = sent_info['gold_sensekeys'][mw_idx]
    if len(set(preds).intersection(set(gold_sensekeys))) > 0:
        result['n_correct'] = 1 # n_correct += 1
        wsd_correct = True
    elif len(preds) > 0:
        result['failed_by_pos'][curr_postag].append((preds[0], gold_sensekeys))
        #failed_by_pos[curr_postag].append((preds[0], gold_sensekeys))
    else:
        result['failed_by_pos'][curr_postag].append((None, gold_sensekeys))
        #failed_by_pos[curr_postag].append((None, gold_sensekeys))

    # register if our prediction belonged to a different POS than gold
    if len(preds) > 0:
        pred_sk_pos = get_sk_pos(preds[0])
        gold_sk_pos = get_sk_pos(gold_sensekeys[0])
        if pred_sk_pos is not None and gold_sk_pos is not None:
            result['pos_confusion'][gold_sk_pos][pred_sk_pos] = 1
        #pos_confusion[gold_sk_pos][pred_sk_pos] += 1

    # register how far the correct prediction was from the top of our matches
    correct_idx = None
    for idx, (matched_sensekey, matched_score) in enumerate(matches):
        if matched_sensekey in gold_sensekeys:
            correct_idx = idx
            result['correct_idxs'].append(idx)
            break

    return result, curr_sense, preds[0] if len(preds) > 0 else None


def eval_batch(batch, sentence_encoder, debugs_f, senses_vsm, args, ft_model):
    batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

    # process contextual embeddings in sentences batches of size args.batch_size
    batch_bert = sentence_encoder.token_embeddings(batch_sents)

    for sent_info, sent_bert in zip(batch, batch_bert):
        debugs_f.write('%s' % sent_info['tokenized_sentence'])
        idx_map_abs = sent_info['idx_map_abs']

        for mw_idx, tok_idxs in idx_map_abs:
            yield eval_mwtok(sent_info, mw_idx, tok_idxs, sent_bert, 
               senses_vsm, args, ft_model)


def run_eval(args):
    """
    Load sense embeddings for evaluation.
    Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
    Load fastText static embeddings if required.
    """
    logging.info('Loading SensesVSM ...')
    if args.sv_path.endswith('.txt') or args.sv_path.endswith('.npz'):
        senses_vsm = SensesVSM(args.sv_path, normalize=True)
    else:
        raise ValueError("" + args.sv_path)
    logging.info('Loaded SensesVSM')

    ft_model = None
    if senses_vsm.ndims in [300+1024, 300+1024+1024]:
        ft_model = load_fast_test(args)

    """
    Initialize various counters for calculating supplementary metrics.
    """
    eval_state = empty_eval_state()

    sentence_encoder =  build_encoder(args)

    """
    Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
    File with predictions is processed by the official scorer after iterating over all instances.
    """
    results_path = 'data/results/%d.%s.%s.key' % (int(time()), args.test_set, args.merge_strategy)
    debugs_path = 'data/results/%d.%s.%s.debug' % (int(time()), args.test_set, args.merge_strategy)
    cnt_sents, total_sents = 0, len(list(testset_generator(args)))
    with open(results_path, 'w') as results_f, open(debugs_path, 'w') as debugs_f:
        for batch_idx, batch in enumerate(chunks(testset_generator(args), args.batch_size)):
            for mwtok_eval, sense, pred in eval_batch(batch, sentence_encoder, 
                                                      debugs_f, senses_vsm, args, ft_model):
                if pred is None:
                    debugs_f.write('%s %s\n' % (sense, 'no-match over thresh'))
                else:
                    results_f.write('%s %s\n' % (sense, pred))
                    debugs_f.write('%s %s\n' % (sense, pred))
            
                eval_state = merge_eval_state(eval_state, mwtok_eval)
                
            if args.debug:
                acc = eval_state['n_correct'] / eval_state['n_instances']
                cnt_sents += len(batch)
                logging.debug('ACC: %.3f (%d %d/%d)' % (
                    acc, eval_state['n_instances'], cnt_sents, total_sents))
            
    if args.debug:
        """
        Summary of supplementary performance metrics.
        """
        logging.info('Supplementary Metrics:')
        logging.info('Avg. correct idx: %.6f' % np.mean(np.array(eval_state['correct_idxs'])))
        logging.info('Avg. correct idx (failed): %.6f' % np.mean(
            np.array([i for i in eval_state['correct_idxs'] if i > 0])))
        logging.info('Avg. num options: %.6f' % np.mean(eval_state['num_options']))
        logging.info('Num. unknown lemmas: %d' % eval_state['n_unk_lemmas'])

        logging.info('POS Failures:')
        for pos, fails in eval_state['failed_by_pos'].items():
            logging.info('%s fails: %d' % (pos, len(fails)))

        logging.info('POS Confusion:')
        for pos in eval_state['pos_confusion']:
            logging.info('%s - %s' % (pos, str(eval_state['pos_confusion'][pos])))

    logging.info('Running official scorer ...')
    run_scorer(args.wsd_fw_path, args.test_set, results_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sv_path', help='Path to sense vectors', required=True)
    parser.add_argument('-ft_path', help='Path to fastText vectors', required=False,
                        default='external/fasttext/crawl-300d-2M-subword.bin')
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-test_set', default='ALL', help='Name of test set', required=False,
                        choices=['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015', 'ALL'])
    parser.add_argument('-min_seq_len', type=int, default=3, help='Minimum sequence length (BERT)', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
    parser.add_argument('-ignore_lemma', dest='use_lemma', action='store_false', help='Ignore lemma features', required=False)
    parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
    parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
    parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
    parser.add_argument('-backend', type=str, default='bert-as-service',
                        help='Underlying BERT model provider',
                        required=False,
                        choices=['bert-as-service', 'transformers'])
    parser.add_argument('-pytorch_model', type=str, default='bert-large-cased',
                        help='Pre-trained transformer name or path',
                        required=False)
    parser.add_argument('-pooling_layer', help='Which layers in the model to take for subtoken embeddings', default=[-4, -3, -2, -1], type=int, nargs='+')
    parser.add_argument('-quiet', dest='debug', action='store_false', help='Less verbose (debug=False)', required=False)
    parser.set_defaults(use_lemma=True)
    parser.set_defaults(use_pos=True)
    parser.set_defaults(debug=True)
    args = parser.parse_args()

    run_eval(args)
