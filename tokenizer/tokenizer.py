import sentencepiece as spm

def tokenize(file):
    path_en = "../data/corpus/" + file + ".en"
    spm.SentencePieceTrainer.Train(
        input=path_en,
        model_prefix="./" + file + "_en",
        vocab_size=16000,
        model_type = "bpe",
        character_coverage = 1.0,
        pad_id = 0,
        unk_id = 1,
        bos_id = 2,
        eos_id = 3,
        shuffle_input_sentence = True,
        num_threads = 16,
        max_sentence_length = 512,
    )

    path_zh = "../data/corpus/" + file + ".zh"
    spm.SentencePieceTrainer.Train(
        input=path_zh,
        model_prefix="./" + file + "_zh",
        vocab_size=16000,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        shuffle_input_sentence=True,
        num_threads=16,
        max_sentence_length=512,
    )

if __name__ == "__main__":
    tokenize("train")