import torch
import sentencepiece as spm

import model
from data import dataset

sp_en = spm.SentencePieceProcessor()
sp_zh = spm.SentencePieceProcessor()

sp_en.Load(model_file="./tokenizer/train_en.model")
sp_zh.Load(model_file="./tokenizer/train_zh.model")

vocab_size_en = sp_en.GetPieceSize()
vocab_size_zh = sp_zh.GetPieceSize()

d_model = 256
num_heads = 8
d_ff = 2048
num_layers = 6
learning_rate = 1e-4

trans_model = model.Transformer(vocab_size_en, vocab_size_zh, d_model, num_heads, d_ff, num_layers)

trans_model.load_state_dict(torch.load("./weights/Transformer_translation.pth"))

trans_model.eval()
with torch.no_grad():
    while True:

        sentence_en = input("请输入一句英文：")
        ids_en = torch.tensor(dataset.encode_line(sentence_en, sp_en), dtype=torch.long)
        ids_en = ids_en.unsqueeze(0)

        sentence_zh = trans_model(ids_en, None, None, None, sp_zh, is_inference=True)
        print(sentence_zh[0])