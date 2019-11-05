import logging
import argparse
import os
from time import time

import lxml.etree
import json
import numpy as np

from sentence_encoder import TransformerSentenceEncoder, min_maxlen_encoder, infer_arch_from_name
import mwtok


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_sense_mapping(eval_path):
    sensekey_mapping = {}
    with open(eval_path) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            sensekey_mapping[id_] = keys
    return sensekey_mapping


def read_xml_sents(xml_path, as_entry_fn=lambda sent_et: sent_et):
    with open(xml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('<sentence '):
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                yield as_entry_fn(lxml.etree.fromstring(''.join(sent_elems)))


def read_xml_sentence(xml_sentence, sense_mapping):
    entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
    for ch in xml_sentence.getchildren():
        for k, v in ch.items():
            entry[k].append(v)
        entry['token_mw'].append(ch.text)

        if 'id' in ch.attrib.keys():
            entry['senses'].append(sense_mapping[ch.attrib['id']])
        else:
            entry['senses'].append(None)

    entry['token'] = sum([t.split() for t in entry['token_mw']], [])
    entry['sentence'] = ' '.join([t for t in entry['token_mw']])
    entry['idx_map_abs'] = mwtok.calc_idx_map_abs(entry['token_mw'])
    return entry
def get_sentence_entry_generator(train_path, eval_path):
    assert train_path.endswith('data.xml')
    sense_mapping = get_sense_mapping(eval_path)
    def sent_et_as_entry(sent_et):
		return read_xml_sentence(sent_et, sense_mapping)
    return read_xml_sents(train_path, as_entry_fn=sent_et_as_entry)


def is_valid_len(sentence, min_len, max_len):
    bert_tokens = bert_tokenizer.tokenize(sentence)
    special_toks = 2
    bl = len(bert_tokens)
    return bl <= (max_len - special_toks) and bl > (min_len - special_toks)


def cosim(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 1, "assumed 1 dim vector, but found " + str(a.shape)
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    return dot / (norma * normb)


def add_sense_vecs(sv, sv_gen, max_instances=float('inf'), max_rolling=100):
    """Adds sense vectors from a generator into sv
    :param sv initial sense_vec instance a map from sense str to `sum_num` maps
    :sv_gen a generator returning tuples `(sense, vec)`"""
    for sense, vec in sv_gen:
        sum_num = sv.get(sense, None)
        if sum_num is None:
            sv[sense] = {'vecs_sum': vec,
                          'vecs_num': 1,
                         'rolling_cosim': []}
        elif sum_num['vecs_num'] + 1 < max_instances:
            curr_avg = sum_num['vecs_sum'] / sum_num['vecs_num']
            if sum_num['vecs_num'] <= max_rolling:
                sum_num['rolling_cosim'] += [cosim(vec, curr_avg)]
            sum_num['vecs_sum'] += vec #sum_num2['vecs_sum']
            sum_num['vecs_num'] += 1
    return sv


def process_entry_batch(batch, sentence_encoder, seg_spec=None):
    """Encodes a batch of sentence entries and extracts sense, vec pairs 
	for the each multiword token in the sentences
    """
    sense_vecs = {}
    batch_sents = [e['sentence'] for e in batch]
    if seg_spec is None:
        batch_bert = sentence_encoder.token_embeddings(batch_sents)
    else:
        (minlen, maxlen, batch_size) = seg_spec
        with min_maxlen_encoder(sentence_encoder, minlen, maxlen) as sent_encoder:
            batch_bert = sent_encoder.token_embeddings(batch_sents)

    for sent_info, sent_bert in zip(batch, batch_bert):
        # handling multi-word expressions, mapping allows matching tokens with mw features
        idx_map_abs = sent_info['idx_map_abs']

        for mw_idx, tok_idxs in idx_map_abs:
            if sent_info['senses'][mw_idx] is None:
                continue

            vec = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float32).mean(axis=0)

            for sense in sent_info['senses'][mw_idx]:
                yield sense, vec


def insert_in_sized_batch(sent_entry, sized_batches, sentence_encoder):
    for seg_spec, batch in sized_batches.items():
        minlen, maxlen, bs = seg_spec
        #logging.info("Insert sent in segment [%d, %d)?" % (minlen, maxlen))
        with min_maxlen_encoder(sentence_encoder, minlen, maxlen) as sent_encoder:
            if sent_encoder.is_valid_len(sent_entry['sentence']):
                #logging.info("... yes")
                batch.append(sent_entry)
                return (minlen, maxlen, bs), batch
            #else:
            #    logging.info("... no")
    # sentence didn't fit into any of the available batches, so return Nones
    return None, None


def train_optimized(train_path, eval_path, sentence_encoder, max_instances=float('inf')):
    """Optimized version of `train`, only works for transformers backend
    The optimization consists in creating batches of similar lengths to avoid having to 
    extract senses for a short sentence using `seq-len` 512.
    bert-as-service won't work for this because you'd need to restart the server before
    executing each individual batch.
    """
    cfg = sentence_encoder.encoder_config
    global_min_len = cfg.get('min_seq_len', 3)
    global_max_len = cfg.get('max_seq_len', 512)
    seg_specs = [(3,    16, 32), (16,   32, 32), #min (exclusive), max, batch_size
                 (32,   64, 16), (64,  128, 16),
                 (128, 256, 16), (256, 512, 8)]
    sized_batches = {(minlen, maxlen, bs): [] for (minlen, maxlen, bs) in seg_specs if not(
        global_min_len > maxlen or
        global_max_len < minlen)}
    sense_vecs, glob_cnt, t0 = {}, 0, time()
    for sent_idx, entry in enumerate(get_sentence_entry_generator(train_path, eval_path)):
        seg_spec, batch = insert_in_sized_batch(entry, sized_batches, sentence_encoder)
        if batch is None:
            continue # skipped entry because of length

        minlen, maxlen, batch_size = seg_spec # batch segment where sentence was put
        glob_cnt += 1
        if len(batch) == batch_size: # sentence completed the batch, process
            sense_vecs = add_sense_vecs(
                sense_vecs, process_entry_batch(batch, sentence_encoder, seg_spec=seg_spec),
                max_instances=max_instances)
            sized_batches[seg_spec] = [] # start new batch

        if glob_cnt % 100 == 0:
            logging.info('%.3f sents/sec - %d sents, %d senses' % (
                    glob_cnt/(time() - t0), glob_cnt, len(sense_vecs)))

    # finished iterating through dataset, but we may have unprocessed examples...
    for seg_spec, batch in sized_batches.items():
        (minlen, maxlen, batch_size) = seg_spec
        if len(batch) > 0:
            logging.info("Processing remaining batch [%d, %d) with %d < %d elts" % (
                minlen, maxlen, len(batch), batch_size))
            sense_vecs = add_sense_vecs(
                sense_vecs, process_entry_batch(batch, sentence_encoder, seg_spec=seg_spec),
                max_instances=max_instances)
    logging.info('#sents: %d of %d, %.3f sents/sec' % (
            glob_cnt, sent_idx,
            glob_cnt/(time() - t0)))
    return sense_vecs


def train(train_path, eval_path, sentence_encoder,
          batch_size=2048,
          max_instances=float('inf')):
    sense_vecs, skipped = {}, 0
    batch, batch_idx, batch_t0 = [], 0, time()
    for sent_idx, entry in enumerate(get_sentence_entry_generator(train_path, eval_path)):
        if sentence_encoder.is_valid_len(entry['sentence']):
            batch.append(entry)
        else:
            skipped += 1
            
        if len(batch) == batch_size:
            batch_sense_vecs = process_entry_batch(batch, sentence_encoder)
            sense_vecs = add_sense_vecs(sense_vecs, batch_sense_vecs,
                                        max_instances=max_instances)
            
            batch_tspan = time() - batch_t0
            logging.info('%.3f sents/sec - %d sents, %d senses' % (
                args.batch_size/batch_tspan, sent_idx, len(sense_vecs)))

            batch, batch_t0 = [], time()
            batch_idx += 1

    logging.info('#sents: %d, skipped (due to length): %d' % (sent_idx, skipped))
    return sense_vecs


def write_out_files(sense_vecs, out_path, args, sep=' '):
    logging.info('Writing Sense Vectors ...')
    vecs_path = out_path[:-3] + "vecs." + out_path[-3:]
    with open(vecs_path, 'w') as vecs_f:
        for sense, vecs_info in sense_vecs.items():
            vec = vecs_info['vecs_sum'] / vecs_info['vecs_num']
            vec_str = sep.join([str(round(v, 6)) for v in vec.tolist()])
            vecs_f.write('%s%s%s\n' % (sense, sep, vec_str))
    logging.info('Written %s' % vecs_path)

    counts_path = out_path[:-3] + "counts." + out_path[-3:]
    logging.info('Writing Sense counts ...')
    with open(counts_path, 'w') as counts_f:
        for sense, vecs_info in sense_vecs.items():
            counts_f.write('%s%s%s\n' % (sense, sep, vecs_info['vecs_num'])) 
    logging.info('Written %s' % counts_path)


    rcosim_path = out_path[:-3] + "rolling_cosims." + out_path[-3:]
    logging.info('Writing rolling cosine similarities ...')
    with open(rcosim_path, 'w') as rcosim_f:
        for sense, vecs_info in sense_vecs.items():
            rcosims = vecs_info['rolling_cosim']
            rcosim_str = sep.join([str(round(v, 6)) for v in rcosims])
            rcosim_f.write('%s%s%s\n' % (sense, sep, rcosim_str)) 
    logging.info('Written %s' % rcosim_path)

    config_path = os.path.dirname(out_path) + '/lmms_config.json'
    logging.info("Writing lmms_train_config.json")
    lmms_config = {'encoder_config': encoder_config(args),
                  'train_args': vars(args)}
    with open(config_path, 'w') as cfg_f:
        json.dump(lmms_config, cfg_f, indent=2)
    logging.info('Written %s' % config_path)


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

def dataset_paths(dataset_name):
    if dataset_name == 'semcor':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
    elif dataset_name == 'semcor_tlgs':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor_14.2_tlgs'
        keys_path = None
    elif dataset_name == 'semcor_tuclgs':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor_14.2_tuclgs'
        keys_path = None
    elif dataset_name == 'semcor_omsti':
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.data.xml'
        keys_path = args.wsd_fw_path + 'Training_Corpora/SemCor+OMSTI/semcor+omsti.gold.key.txt'
    elif dataset_name == 'europarl_14.2_tuclgs':
        train_path = '/data/corpora/europarlv7/es-en/en/seq/14.2_tuclgs'
        keys_path = None
    return train_path, keys_path

def run_train(args):
    (dir, fname) = os.path.split(args.out_path)
    if not os.path.isdir(dir):
        raise ValueError("out path doesn't exist %s" % args.out_path)

    train_path, keys_path = dataset_paths(args.dataset)
    assert args.out_path[-4:] in ['.txt', '.tsv'], "out_path must end in .txt or .tsv " + args.out_path 
        
    sentence_encoder =  build_encoder(args)
    if args.backend == 'bert-as-service':
        sense_vecs = train(train_path, keys_path,
                       sentence_encoder,
                       batch_size=args.batch_size,
                       max_instances=args.max_instances)
    elif args.backend == 'transformers':
        sense_vecs = train_optimized(train_path, keys_path, 
                                     sentence_encoder, 
                                     max_instances=args.max_instances)

    sep = ' '
    if args.out_path.endswith('.tsv'):
        sep='\t'
    out_path = "%s.%d-%d.%s" % (args.out_path[:-3], # root file
                                args.min_seq_len, args.max_seq_len, # seq lens
                                args.out_path[-3:]) # file extension
    write_out_files(sense_vecs, out_path, args, sep=sep)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Initial Sense Embeddings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='external/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=False,
                        choices=['semcor', 'semcor_omsti'])
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-min_seq_len', type=int, default=3, help='Minimum sequence length (BERT)', required=False)
    parser.add_argument('-max_seq_len', type=int, default=512, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-max_instances', type=float, default=float('inf'), help='Maximum number of examples for each sense', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=True)
    parser.add_argument('-pooling_layer', help='Which layers in the model to take for subtoken embeddings', default=[-4, -3, -2, -1], type=int, nargs='+')
    parser.add_argument('-backend', type=str, default='bert-as-service',
                        help='Underlying BERT model provider',
                        required=False,
                        choices=['bert-as-service', 'transformers'])
    parser.add_argument('-pytorch_model', type=str, default='bert-large-cased',
                        help='Pre-trained transformer name or path',
                        required=False)
    args = parser.parse_args()
    run_train(args)
