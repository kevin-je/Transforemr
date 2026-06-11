import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import utils

def read_lines(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def encode_line(line, sp):
    _id = sp.EncodeAsIds(line)
    _id = [sp.bos_id()] + _id + [sp.eos_id()]
    return _id


class TranslateDataset(Dataset):
    def __init__(self, path_en, path_zh, sp1, sp2):
            self.ids_en = [torch.tensor(encode_line(line, sp1), dtype=torch.long) for line in read_lines(path_en)]
            self.ids_zh = [torch.tensor(encode_line(line, sp2), dtype=torch.long) for line in read_lines(path_zh)]

    def __len__(self):
        return len(self.ids_en)

    def __getitem__(self, item):
        return self.ids_en[item], self.ids_zh[item]




def collate_fn(batch):
    inputs, labels = zip(*batch)

    # inputs = [x[:50] for x in inputs]
    # labels = [y[:50] for y in labels]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    tgt_in = labels[:, :-1]
    tgt_out = labels[:, 1:]

    src_mask = utils.get_src_mask(inputs)
    tgt_mask = utils.get_tgt_mask(tgt_in)

    return inputs, tgt_in, tgt_out, src_mask, tgt_mask