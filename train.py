import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sentencepiece as spm
from sacrebleu.metrics import BLEU

import model
from data import dataset
import utils

# 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 32
d_model = 256
num_heads = 8
d_ff = 2048
num_layers = 6
learning_rate = 1e-4
max_len = 50

save_dir = "/root/autodl-tmp/weights"

# 数据集路径
dataset_train_en = "./data/corpus/train.en"
dataset_train_zh = "./data/corpus/train.zh"
dataset_val_en = "./data/corpus/dev.en"
dataset_val_zh = "./data/corpus/dev.zh"

sp_en = spm.SentencePieceProcessor()
sp_zh = spm.SentencePieceProcessor()

if __name__ == "__main__":

    sp_en.Load(model_file="./tokenizer/train_en.model")
    sp_zh.Load(model_file="./tokenizer/train_zh.model")

    vocab_size_en = sp_en.GetPieceSize()
    vocab_size_zh = sp_zh.GetPieceSize()

    model = model.Transformer(
        vocab_size_en, vocab_size_zh, d_model, num_heads, d_ff, num_layers
    ).to(device)
    model.load_state_dict(torch.load("weights/Transformer_translation.pth"))

    train_dataset = dataset.TranslateDataset(dataset_train_en, dataset_train_zh, sp_en, sp_zh)
    val_dataset = dataset.TranslateDataset(dataset_val_en, dataset_val_zh, sp_en, sp_zh)

    dataloader_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=dataset.collate_fn
    )
    dataloader_val = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, collate_fn=dataset.collate_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bleu_score_lst = []
    best_bleu_score = 0

    for epoch in range(epochs):
        # 训练模型
        model.train()
        print("=" * 6 + f"当前轮次：{epoch + 1}" + "=" * 6)
        loss_sum = 0

        for batch_idx, (inputs, tgt_in, tgt_out, src_mask, tgt_mask) in enumerate(dataloader_train):
            inputs = inputs.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            logits = model(inputs, tgt_in, src_mask, tgt_mask, sp_zh)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 1000 == 999:
                print(f"批次{batch_idx + 1}已完成！损失为：{loss.item():.2f}")

        loss_avg = loss_sum / len(dataloader_train)
        print(f"---当前轮次的训练集平均损失：{loss_avg:.2f}---")

        # 测试模型
        model.eval()
        with torch.no_grad():
            total_sentences_c = []
            total_sentences_r = []
            for batch_idx, (inputs, _, tgt_out, src_mask, _) in enumerate(dataloader_val):
                inputs = inputs.to(device)
                tgt_out = tgt_out.to(device)
                src_mask = src_mask.to(device)

                sentences_c = model(inputs, None, src_mask, None, sp_zh, max_len=max_len, is_inference=True)
                sentences_r = [utils.tokens_to_str(sentence_r, sp_zh) for sentence_r in tgt_out]

                total_sentences_c.extend(sentences_c)
                total_sentences_r.extend(sentences_r)

            bleu = BLEU(tokenize="zh")
            bleu_score = bleu.corpus_score(total_sentences_c, [total_sentences_r]).score
            bleu_score_lst.append(bleu_score)
            print(f"---当前轮次的验证集BLEU指标：{bleu_score:.2f}%---")

            # 保存模型参数
            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score

                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{save_dir}/Transformer_translation.pth")

    # 绘制训练图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x_date = np.arange(epochs) + 11
    y_data = np.array(bleu_score_lst)
    plt.xlabel("训练轮次")
    plt.ylabel(r"BLEU（$\text{%}$）")
    plt.title("Transformer在英译中数据集上的BLEU指标")
    plt.grid(True)
    plt.plot(x_date, y_data, marker="o", color="r")
    plt.savefig("Transformer_translation17-48.png")
    plt.show()