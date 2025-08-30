from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def prepare_tokenizer(unk_token, special_tokens):
    tokenizer = Tokenizer(WordLevel(unk_token = unk_token))
    trainer = WordLevelTrainer(special_tokens = special_tokens)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single = '[CLS] $A [SEP]',
        special_tokens = [('[CLS]', 1), ('[SEP]', 2)]
    )
    return tokenizer, trainer

def train_tokenizer(tokenizer, trainer, data_file_name: str, tokenizer_file_name: str, special_tokens_dict: dict):
    tokenizer.train([data_file_name], trainer)
    tokenizer.save(tokenizer_file_name)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file_name)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def tokenize(sentence, tok):
        return tok(sentence['text'], max_length=128, truncation=True, padding=True)