from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
from data.mt_dataset import MTDataset
from data.bpe_tokenizer import BPETokenizer
from data.utils import TextUtils, short_text_filter_function


class DataManager(LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None
        self.device = device

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"], shuffle=False)
        prefix_filter = self.config['prefix_filter']
        if prefix_filter:
            prefix_filter = tuple(prefix_filter)

        source_sentences, target_sentences = [], []
        # dataset is ambiguous -> i lied -> я солгал/я соврала
        unique_sources = set()
        for pair in pairs:
            source, target = pair[0], pair[1]
            if short_text_filter_function(pair, self.config['max_length'], prefix_filter) and source not in unique_sources:
                source_sentences.append(source)
                target_sentences.append(target)
                unique_sources.add(source)
                
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

        train_size = int(len(source_sentences)*self.config["train_size"])
        source_train_sentences, source_val_sentences = source_sentences[:train_size], source_sentences[train_size:]
        target_train_sentences, target_val_sentences = target_sentences[:train_size], target_sentences[train_size:]

        # TODO: Замените на BPE токенизатор DONE
        self.source_tokenizer = BPETokenizer(source_train_sentences, self.config['max_length'], pad_flag=True)
        
        tokenized_source_train_sentences = [self.source_tokenizer(s) for s in source_train_sentences]
        tokenized_source_val_sentences = [self.source_tokenizer(s) for s in source_val_sentences]

        # TODO: Замените на BPE токенизатор DONE
        self.target_tokenizer = BPETokenizer(target_train_sentences, self.config['max_length'], pad_flag=True)

        tokenized_target_train_sentences = [self.target_tokenizer(s) for s in target_train_sentences]
        tokenized_target_val_sentences = [self.target_tokenizer(s) for s in target_val_sentences]

        self.train_dataset = MTDataset(tokenized_source_list=tokenized_source_train_sentences,
                                  tokenized_target_list=tokenized_target_train_sentences, dev=self.device)

        self.val_dataset = MTDataset(tokenized_source_list=tokenized_source_val_sentences,
                                tokenized_target_list=tokenized_target_val_sentences, dev=self.device)
        
        self.input_lang_n_words  = len(self.source_tokenizer.word2index)
        self.output_lang_n_words = len(self.target_tokenizer.word2index)
        
        return self.train_dataloader, self.val_dataloader

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.config["batch_size"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.config["batch_size"], drop_last=True )
