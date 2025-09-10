import argparse, os, sys

from transformers import (
    AutoTokenizer, 
    BertForMaskedLM
)

import numpy as np
from scipy.spatial import distance
import torch

#from tf.fabric import Fabric
#SYR = Fabric(locations='~/data/tf_data/syriac/0.2')
#api = SYR.load('''
#    otype sp vt sp vt g_pfm g_pfx g_vbs lex g_vbe g_nme g_emf ls
#''')
from tf.app import use
SYR = use('etcbc/syriac')

Fsyr, Lsyr, Tsyr = SYR.api.F, SYR.api.L, SYR.api.T
Fsyr.dataset = 'syriac'

#MT = Fabric(locations='~/data/tf_data/hebrew/2021')
#api = MT.load('''
#        otype vt sp g_lex_utf8 g_prs_utf8 g_nme_utf8 g_pfm_utf8 g_vbs_utf8 g_vbe_utf8 g_uvf_utf8 nametype
#       ''')
#Fheb, Lheb, Theb = MT.api.F, MT.api.L, MT.api.T
MT = use('etcbc/bhsa', version='2021')
MT.load(['g_lex_utf8', 'g_prs_utf8', 'g_nme_utf8', 'g_pfm_utf8', 'g_vbs_utf8', 'g_vbe_utf8', 'g_uvf_utf8'])
Fheb, Lheb, Theb = MT.api.F, MT.api.L, MT.api.T

Fheb.dataset = 'hebrew' 

import bert_data_preprocessing_functions as bf

MODEL_FOLDER = '../models/'
seq_length = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name,
                                              return_dict_in_generate=True, 
                                              output_hidden_states=True).to(device)

    model.eval()
    return tokenizer, model

def get_hidden_states(heb_texts_dict, model, tokenizer):
    hidden_states = {}
    for key, text_chunk in heb_texts_dict.items():
        tokenized_inputs = tokenizer(text_chunk, max_length=128, truncation=True, padding=True, return_tensors="pt")
        tokenized_inputs = {k:v.to(device) for k,v in tokenized_inputs.items()}
        
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            last_hidden_states = outputs.hidden_states[-1].cpu().numpy()
            hidden_states[key] = last_hidden_states

    return hidden_states

def process_hidden_states(hidden_states, approach):
    processed_hidden_states = {}

    for key, hs in hidden_states.items():
        if approach == 'mean':
            state= np.mean(hs, 1)
        elif approach == 'cls':
            state = hs[:, 0, :]
        state = np.squeeze(state)
        processed_hidden_states[key] = state 
    return processed_hidden_states

def get_keys(hebrew_bible, target_book):
    target_book_keys = [key for key in hebrew_bible.n_clause_dict.keys() if target_book in key[0]]
    non_target_book_keys = [key for key in hebrew_bible.n_clause_dict.keys() if target_book not in key[0]]

    return target_book_keys, non_target_book_keys

def make_heb_syr_cosine_distance_array(book, embeddings_dict_heb, embeddings_dict_syr):

    # Remove '1_' and '2_' from booknames 1_Chronicles, 2 Chronicles, etc.
    #if '_' in book:
    #    book = book.split('_')[1]

    cosine_dists = np.zeros((len(embeddings_dict_heb), len(embeddings_dict_syr)))
    for heb_idx, heb_embedding in enumerate(embeddings_dict_heb.values()):
        for syr_idx, syr_embedding in enumerate(embeddings_dict_syr.values()):
            cosine_dists[heb_idx, syr_idx] = distance.cosine(heb_embedding, syr_embedding)
    return cosine_dists

def make_min_dist_dict(distances, heb_keys, syr_keys):

    min_dist_dict = {}
    min_indices = np.argmin(distances, axis=1)
    for heb_idx, syr_idx in enumerate(min_indices):
        heb_key = heb_keys[heb_idx]
        syr_key = syr_keys[syr_idx]
        min_dist_dict[heb_key] = syr_key

    return min_dist_dict

def calculate_model_performance(min_dist_dict: dict):
    correct = 0
    wrong = 0
    for heb_key, syr_value in min_dist_dict.items():
        if heb_key == syr_value:
            correct += 1
        else:
            wrong += 1

    return correct, wrong

def make_top_n_indices_array(distances_array, n):
    top_n_min_indices = np.argpartition(distances_array, n, axis=1)[:, :n]
    return top_n_min_indices

def calculate_top_n_model_performance(top_n_min_indices, syr_keys, heb_keys):

    in_top_n_count = 0
    for i in range(top_n_min_indices.shape[0]):
        top_idcs = top_n_min_indices[i]
        top_n_syr_keys = [syr_keys[idx] for idx in top_idcs]

        true_key = heb_keys[i]
        if true_key in top_n_syr_keys:
            in_top_n_count += 1
    return in_top_n_count

def get_target_book_keys(hebrew_bible):
    target_book_keys = [key for key in hebrew_bible.n_clause_dict.keys() if target_book in key[0]]
    return target_book_keys
    
def evaluate_model(args): #model_name: str, target_book):

    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', metavar='model_name', help='Name of the model that is trained on Hebrew and Syriac data', type=str)
    parser.add_argument('-bo', metavar='test_book', help='Name of the biblical book on which the model is tested.', type=str, default='Genesis')
    args = parser.parse_args()
    
    hebrew_text, syriac_text = bf.prepare_hebrew_and_syriac_verse_data(
        True,
        seq_length,
        Fheb, Lheb, Theb,
        Fsyr, Lsyr, Tsyr
    )
    
    eval_dict = {}

    print(f'Model name: {args.mo}')
    print('Get text representation')

    hebrew_morphemes_target_book = {k:v for (k, v) in hebrew_text.items() if k[0] == args.bo}
    target_book_keys = [key for key in hebrew_text.keys() if key[0] == args.bo]

    syriac_morphemes = {k:v for (k, v) in syriac_text.items()} # if k[0] == target_book}

    print('Load model and tokenizer')
    model_path = MODEL_FOLDER + args.mo
    tokenizer, model = load_model_and_tokenizer(model_path)

    print('Retrieve hidden states from model')
    hidden_states_hebrew = get_hidden_states(hebrew_morphemes_target_book, model, tokenizer)
    hidden_states_syriac = get_hidden_states(syriac_morphemes, model, tokenizer)
    
    for embedding_method in ['mean', 'cls']:
        print(f'Process hidden states with {embedding_method}')
        processed_hidden_states_hebrew = process_hidden_states(hidden_states_hebrew, embedding_method)
        processed_hidden_states_syriac = process_hidden_states(hidden_states_syriac, embedding_method)

        print(f'Calculate cosine distances using {embedding_method}')
        distances_array = make_heb_syr_cosine_distance_array(args.bo, processed_hidden_states_hebrew, processed_hidden_states_syriac)
        heb_keys = list(hebrew_morphemes_target_book.keys())
        syr_keys = list(syriac_morphemes.keys())

        top_1_min_indices = make_top_n_indices_array(distances_array, 1)
        in_top_1_count = calculate_top_n_model_performance(top_1_min_indices, syr_keys, heb_keys)

        top_5_min_indices = make_top_n_indices_array(distances_array, 5)
        in_top_5_count = calculate_top_n_model_performance(top_5_min_indices, syr_keys, heb_keys)

        top_10_min_indices = make_top_n_indices_array(distances_array, 10)
        in_top_10_count = calculate_top_n_model_performance(top_10_min_indices, syr_keys, heb_keys)

        print(f'Evaluation {embedding_method}')
        print(f'Evaluation of {len(heb_keys)} verses.')

        print(f'Completely correct: {in_top_1_count / len(heb_keys)}' )
        print(f'In top 5: {in_top_5_count / len(heb_keys)}' )
        print(f'In top 10: {in_top_10_count / len(heb_keys)}' )
        print()
        
    #return distances_array
    
    
#model_name = 'C:/Users/geitb/Zurich/zurich_research/models_from_cluster/models/morphs_marks_lex_6_lay_8_atth_0.0001_lr_5_sl_0.002_hebap_1_syr_0.01_syrap'


if __name__ == "__main__":
    evaluate_model(sys.argv[1:])
    #distances_array = evaluate_model(model_name, heb_morpheme_dataset, syr_morpheme_dataset, 'Genesis')
