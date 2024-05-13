from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from pytorch_lightning import LightningDataModule
from data.mt_dataset import MTDataset
from data.utils import TextUtils, short_text_filter_function


class DataManager(LightningDataModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.input_lang_n_words = None
        self.output_lang_n_words = None
        self.device = device

    def prepare_data(self):
        pairs = TextUtils.read_langs_pairs_from_file(filename=self.config["filename"])
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

        #######################################################################################
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        task_prefix = "translate English to Russian: "
        
        #######################################################################################
        self.source_train_encoding = self.tokenizer(
            [task_prefix + sequence for sequence in source_train_sentences],
            padding="longest",
            max_length=self.config['max_length'],
            truncation=False,
            return_tensors="pt",
        )
        
        self.source_val_encoding = self.tokenizer(
            [task_prefix + sequence for sequence in source_val_sentences],
            padding="longest",
            max_length=self.config['max_length'],
            truncation=False,
            return_tensors="pt",
        )
        
        #######################################################################################
        self.target_train_encoding = self.tokenizer(
            target_train_sentences,
            padding="longest",
            max_length=self.config['max_length'],
            truncation=True,
            return_tensors="pt",
        )
        
        self.target_val_encoding = self.tokenizer(
            target_val_sentences,
            padding="longest",
            max_length=self.config['max_length'],
            truncation=True,
            return_tensors="pt",
        )

        self.train_dataset = MTDataset(tokenized_source_list=self.source_train_encoding.input_ids,
                                       attention_mask_source_list=self.source_train_encoding.attention_mask,
                                       tokenized_target_list=self.target_train_encoding.input_ids, dev=self.device)

        self.val_dataset = MTDataset(tokenized_source_list=self.source_val_encoding.input_ids,
                                     attention_mask_source_list=self.source_val_encoding.attention_mask,
                                     tokenized_target_list=self.target_val_encoding.input_ids, dev=self.device)
        
        return self.train_dataloader, self.val_dataloader

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=True )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=True, batch_size=self.config["batch_size"], drop_last=True )
