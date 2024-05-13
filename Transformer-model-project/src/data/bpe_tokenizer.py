from tokenizers import Tokenizer
import torch
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    def __init__(self, sentence_list, max_sent_length, pad_flag):
        """
        sentence_list - список предложений для обучения
        """
        self.pad_flag = pad_flag
        self.special_tokens_set = ["SOS", "EOS", "UNK", "PAD"]
        self.max_sent_length = max_sent_length
        
        self.tokenizer = Tokenizer(BPE(unk_token="UNK"))
        trainer = BpeTrainer(special_tokens=self.special_tokens_set)
        
        self.tokenizer.pre_tokenizer = Whitespace()  
        self.tokenizer.train_from_iterator(sentence_list, trainer)
         
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {index: word for word, index in self.word2index.items()}

        self.tokenizer.post_processor = TemplateProcessing(
            single="SOS $A EOS",
            special_tokens=[
                ("SOS", self.word2index["SOS"]),
                ("EOS", self.word2index["EOS"]),
            ],
        )

        print(f'Space tokenizer fitted - {len(self.word2index)} tokens')
    
    def __call__(self, sentence):
        encoded_sentence = self.tokenizer.encode(sentence).ids

        if self.pad_flag and len(encoded_sentence) < self.max_sent_length:
            padding_length = self.max_sent_length - len(encoded_sentence)
            encoded_sentence += [self.word2index['PAD']] * padding_length
        else:
            encoded_sentence = encoded_sentence[:self.max_sent_length - 1] + [self.word2index['EOS']]
        
        return encoded_sentence

    def decode(self, token_list: torch.Tensor):
        return self.tokenizer.decode(token_list.tolist()).split(' ')
