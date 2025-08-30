import pandas as pd

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback,
    BertForMaskedLM,
    Trainer
)

from tf.app import use
SYR = use('etcbc/syriac')
Fsyr, Lsyr, Tsyr = SYR.api.F, SYR.api.L, SYR.api.T
Fsyr.dataset = 'syriac'

MT = use('etcbc/bhsa')
MT.load([
        'g_lex_utf8', 'g_prs_utf8', 'g_nme_utf8', 'g_pfm_utf8', 'g_vbs_utf8', 'g_vbe_utf8', 'g_uvf_utf8'
       ])
Fheb, Lheb, Theb = MT.api.F, MT.api.L, MT.api.T
Fheb.dataset = 'hebrew' 

import bert_data_preprocessing_functions as bf
import bert_model_details as bmd
import tokenizer_functions as tokf

data_file_name = 'hebrew_bible_morphemes.txt'
special_tokens_dict = {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
tokenizer_file_name = 'morphemes_with_markers_tokenizer.json'

# model details
seq_length = 5
num_hidden_layers = 6
num_attention_heads = 8
learning_rate=0.0001
num_epochs = 150

augment_hebrew = True
augment_hebrew_prob = 0.001
add_syriac= True
augment_syriac = False
augment_syriac_prob = 0 #0.005

k_folds = 10

    
def main(augment_hebrew=False, augment_hebrew_prob=0, add_syriac=False, augment_syriac=False, augment_syriac_prob=0):

    morpheme_dataset = bf.prepare_hebrew_and_syriac_data(augment_hebrew, 
                                                      augment_hebrew_prob, 
                                                      add_syriac, 
                                                      augment_syriac, 
                                                      augment_syriac_prob,
                                                      seq_length,
                                                      data_file_name,
                                                      k_folds,
                                                      Fheb, Lheb, Theb,
                                                      Fsyr, Lsyr, Tsyr
                                                      )

    print('Make Dataset object')
    bib_df = pd.Series(morpheme_dataset).to_frame('text')
    bib_df_shuffled = bib_df.sample(frac=1, replace=False, ignore_index=True)
    bib_ds = Dataset.from_pandas(bib_df_shuffled)

    print('Prepare tokenizer')
    special_tokens = list(special_tokens_dict.values())
    tokenizer, trainer = tokf.prepare_tokenizer(special_tokens_dict['unk_token'], special_tokens)
    tokenizer = tokf.train_tokenizer(tokenizer, trainer, data_file_name, tokenizer_file_name, special_tokens_dict)

    print('Tokenize data')
    tokenized_data = bib_ds.map(tokf.tokenize, fn_kwargs={'tok': tokenizer}, batched=True)
    tokenized_data.set_format("pt", columns=['input_ids', 'attention_mask'], output_all_columns=True)
    tokenized_data = tokenized_data.train_test_split(test_size=0.1)

    print('Initialize model')
    model_name = bmd.make_model_name(num_hidden_layers, num_attention_heads, learning_rate, seq_length, augment_hebrew_prob, add_syriac, augment_syriac_prob)
    print(model_name)
    model_details = bmd.ModelDetails(num_hidden_layers, num_attention_heads, learning_rate, num_epochs, model_name, tokenizer)
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased', config=model_details.config, ignore_mismatched_sizes=True)
    model =  model_details.randomize_model(model)

    print('Train model')
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=.15)
    trainer = Trainer(
        model=model,
        args=model_details.args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
      )
    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    main(augment_hebrew, augment_hebrew_prob, add_syriac, augment_syriac, augment_syriac_prob)