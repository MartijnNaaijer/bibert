from transformers import (
    BertConfig, 
    TrainingArguments
)
import torch


def make_model_name(num_hidden_layers, num_attention_heads, learning_rate, seq_length, hebrew_augm_prob, add_syriac, syriac_augm_prob):
    syriac_data_added = 0
    if add_syriac:
        syriac_data_added = 1
    if not add_syriac:
        syriac_augm_prob = 0
    return f'morphs_marks_lex_{num_hidden_layers}_lay_{num_attention_heads}_atth_{learning_rate}_lr_{seq_length}_sl_{hebrew_augm_prob}_hebap_{syriac_data_added}_syr_{syriac_augm_prob}_syrap'

class ModelDetails:
    def __init__(self, num_hidden_layers: int, num_attention_heads: int, learning_rate: float, num_epochs: int, model_name: str, tokenizer):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_epochs = num_epochs
        self.model_name: str = model_name
        
        self.args = TrainingArguments(output_dir=self.model_name, 
                         save_strategy='epoch',
                         eval_strategy = 'epoch',
                         learning_rate=learning_rate,
                         num_train_epochs=num_epochs,
                         per_device_train_batch_size=8, 
                         per_device_eval_batch_size=8,
                         load_best_model_at_end=True,
                         metric_for_best_model='eval_loss',
                         greater_is_better=False,
                         save_safetensors=False,
                         seed=42,
                        )

        self.config = BertConfig.from_pretrained(
                          'bert-base-multilingual-cased', 
                           model_type='bert',
                           attention_probs_dropout_prob=.5, 
                           hidden_dropout_prob=.5, 
                           hidden_size=256,
                           intermediate_size=1024,
                           max_position_embeddings=128,
                           num_attention_heads=num_attention_heads,
                           num_hidden_layers=num_hidden_layers,
                           vocab_size=len(tokenizer.vocab)
                           )
        
    @staticmethod    
    def randomize_model(model):
        for module_ in model.named_modules(): 
            if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
                module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif isinstance(module_[1], torch.nn.LayerNorm):
                module_[1].bias.data.zero_()
                module_[1].weight.data.fill_(1.0)
            if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
                module_[1].bias.data.zero_()
        return model