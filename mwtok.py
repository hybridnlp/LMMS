
def calc_idx_map_abs(tokens_mw):
    """Calculate a mapping from multiword token ids to whitespace token ids

    This allows handling of multi-word expressions as the mapping allows
    matching whitespace tokens with mw features

    For example:
      calc_idx_map_abs(["single", "multi word", "token"])
    will return:
      [[0, [0]], [1, [1, 2]], [2, [3]]]

    :param tokens_mw: a list of multiword tokens 
    :returns: a list of list each sublist contains the id of the multiword and a
      list of ids for the corresponding whitespace tokens
    :rtype: list

    """
    idx_map_abs = []
    idx_map_rel = [(i, list(range(len(t.split()))))
                   for i, t in enumerate(tokens_mw)]
    token_counter = 0
    for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
        idx_tokens = [i+token_counter for i in idx_tokens]
        token_counter += len(idx_tokens)
        idx_map_abs.append([idx_group, idx_tokens])
    return idx_map_abs
