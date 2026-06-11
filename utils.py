import torch

def get_src_mask(inputs):
    return (inputs != 0).unsqueeze(1).unsqueeze(2)

def get_tgt_mask(tgt_in):
    seq_len = tgt_in.size(1)

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt_in.device), diagonal=1).bool()
    causal_mask = ~causal_mask

    padding_mask = get_src_mask(tgt_in)

    return causal_mask & padding_mask

def strip_special_tokens(ids):
    stripped_ids = []
    for _id in ids:
        _id = int(_id)
        if _id == 3:
            break
        if _id in (0, 2):
            continue
        stripped_ids.append(_id)
    return stripped_ids

def tokens_to_str(ids, sp):
    ids = strip_special_tokens(ids)
    return sp.DecodeIds(ids)