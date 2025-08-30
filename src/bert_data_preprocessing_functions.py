import itertools, random


class GeneralData:
    """Class containing data that Hebrew and Syriac have in common.
    Hebrew morpheme markers
    nme_marker =  '֜'
    pfm_marker =  'ְ'
    vbs_marker =  'ֱ'
    vbe_marker =  'ֲ'
    prs_marker =  'ֳ'
    uvf_marker =  'ִ'
    
    Syriac morpheme markers
    pfx_marker =  '֚'
    emf_marker =  'ֽ'
    """
    def __init__(self):
        self.relevant_chars_utf8: set = {' ', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח',
                                         'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ',
                                         'ס', 'ע', 'ף', 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת'}
 
        self.heb_morph_marker_dict: dict = {
                                   4:  '֜',
                                   0:  'ְ',
                                   1:  'ֱ',
                                   3:  'ֲ',
                                   6:  'ֳ',
                                   5:  'ִ'
                                   }
        self.syr_morph_marker_dict: dict = {
                             5:  '֜',
                             0:  'ְ',
                             1:  '֚',
                             2:  'ֱ',
                             4:  'ֲ',
                             6:  'ֽ',
                             7:  'ֳ'
        }
        # needed for conversion of lex (ASCII in TF) to hebrew script, including markers =, / and [
        self.alphabet_dict_heb_lat:dict = {'א': '>',
                                      'ב': 'B',
                                      'ג': 'G',
                                      'ד': 'D',
                                      'ה': 'H',
                                      'ו': 'W',
                                      'ז': 'Z',
                                      'ח': 'X',
                                      'ט': 'V',
                                      'י': 'J',
                                      'כ': 'K',
                                      'ל': 'L',
                                      'מ': 'M',
                                      'נ': 'N',
                                      'ס': 'S',
                                      'ע': '<',
                                      'פ': 'P',
                                      'צ': 'Y',
                                      'ק': 'Q',
                                      'ר': 'R',
                                      'ש': 'C',
                                      'ת': 'T'
                                      }
        self.alphabet_dict_lat_heb: dict = {v:k for k,v in self.alphabet_dict_heb_lat.items()}
        self.alphabet_dict_lat_heb['_'] = ' '
        self.alphabet_dict_lat_heb['F'] = 'ש' + 'ׂ'
        self.alphabet_dict_lat_heb['/'] = 'ֶ' # nouns/adjectives
        self.alphabet_dict_lat_heb['['] = 'ַ' # verbs
        self.alphabet_dict_lat_heb['='] = 'ֻ' # lex disambiguation marker


class HebrewWord:
    def __init__(self, F, w, relevant_chars_utf8, convert_ascii_string_to_heb_script):

        self.sp: str = F.sp.v(w)

        self.pfm_utf8 = ' '.join([char for char in F.g_pfm_utf8.v(w) if char in relevant_chars_utf8])
        self.vbs_utf8 = ' '.join([char for char in F.g_vbs_utf8.v(w) if char in relevant_chars_utf8])
        self.lex_rep = F.lex.v(w)
        self.lex_rep = convert_ascii_string_to_heb_script(self.lex_rep)
        self.vbe_utf8 = ' '.join([char for char in F.g_vbe_utf8.v(w) if char in relevant_chars_utf8])
        self.nme_utf8 = ' '.join([char for char in F.g_nme_utf8.v(w) if char in relevant_chars_utf8])
        self.uvf_utf8 = ' '.join([char for char in F.g_uvf_utf8.v(w) if char in relevant_chars_utf8])
        self.prs_utf8 = ' '.join([char for char in F.g_prs_utf8.v(w) if char in relevant_chars_utf8])

        self.morph_list = [self.pfm_utf8, self.vbs_utf8, self.lex_rep, self.vbe_utf8, self.nme_utf8, self.uvf_utf8, self.prs_utf8]

    
class SyriacWord:
    def __init__(self, F, w, relevant_chars_utf8, convert_ascii_string_to_heb_script):

        self.sp: str = F.sp.v(w)
        
        self.pfm: str = F.g_pfm.v(w)
        self.pfx: str = F.g_pfx.v(w)
        self.vbs: str = F.g_vbs.v(w)
        self.lex: str = F.lex.v(w)
        self.lex: str = self.make_lex_representation_with_verb_noun_marker(self.lex)
        self.vbe: str = F.g_vbe.v(w)
        self.nme: str = F.g_nme.v(w)
        self.emf: str = F.g_emf.v(w)

        self.pfm_utf8: str = convert_ascii_string_to_heb_script(self.pfm)
        self.pfx_utf8: str = convert_ascii_string_to_heb_script(self.pfx)
        self.vbs_utf8: str = convert_ascii_string_to_heb_script(self.vbs)
        self.lex_utf8: str = convert_ascii_string_to_heb_script(self.lex)
        self.vbe_utf8: str = convert_ascii_string_to_heb_script(self.vbe)
        self.nme_utf8: str = convert_ascii_string_to_heb_script(self.nme)
        self.emf_utf8: str = convert_ascii_string_to_heb_script(self.emf)

        self.morph_list = [self.pfm_utf8, self.pfx_utf8, self.vbs_utf8, self.lex_utf8, self.vbe_utf8, self.nme_utf8, self.emf_utf8]

    def make_lex_representation_with_verb_noun_marker(self, lex):
        if self.sp == 'verb':
            lex += '['
        elif self.sp in {'subs', 'adjv', 'nmpr'}:
            lex += '/'
        if lex == '=':
            lex = ''
        return lex
    

class TFNodePreparator:
    def __init__(self, n, otype, F, L, T):
        """
        n: int number of clauses (Hebrew) in a sequence. We use words for syriac: n * 5
        k: int number of folds in which the dataset is split. Each fold contains 1 / k part of all the chapters. 
        otype: str Type of objects in a sequence. Has value 'word' (Syriac) or 'clause' (Hebrew).
        
        """
        self.n = n 
        self.otype = otype 
        self.F = F 
        self.L = L 
        self.T = T
        
    def split_chapter_list_in_k_equal_lists(self, p):
        chapter_nodes = list(self.F.otype.s('chapter'))
        random.seed(10)
        random.shuffle(chapter_nodes)
        k, m = divmod(len(chapter_nodes), p)
        return list(chapter_nodes[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(p))

    def make_clause_list(self, tf_book_node: int, test_chapters):
        chapters = self.L.d(tf_book_node, 'chapter')
        train_chapters = [ch for ch in chapters if ch not in test_chapters]
        clause_lists = [self.L.d(ch, 'clause') for ch in train_chapters]
        clause_nodes = list(itertools.chain.from_iterable(clause_lists))
        return clause_nodes
    
    @staticmethod
    def save_test_chapter_ids(test_chapters):
        with open('test_chapter_nodes.txt', 'w') as f:
            for chapter_id in test_chapters:
                f.write(f'{str(chapter_id)}\n')

    def make_non_overlapping_n_grams(self, input_list):
        return [input_list[i:i+self.n] for i in range(0, len(input_list), self.n)]

    def make_words_n_gram_dict(self, test_chapters=[]):
        """
        
        """
        n_clause_dict = {}

        for bo in self.F.otype.s('book'):
            if self.F.dataset == 'hebrew':
                clause_nodes = self.make_clause_list(bo, test_chapters)
                n_grams = list(self.make_non_overlapping_n_grams(clause_nodes))
            elif self.F.dataset == 'syriac':
                n_grams = list(self.make_non_overlapping_n_grams(self.L.d(bo, self.otype)))
        
            for n_gram in n_grams:
                ch = self.L.u(n_gram[0], 'chapter')[0]
                book, chapter_number = self.T.sectionFromNode(ch)
                
                if self.F.dataset == 'hebrew':
                    words_n_clause = sorted(list(itertools.chain(*[self.L.d(cl, 'word') for cl in n_gram])))
                    n_clause_dict[(book, chapter_number, tuple(words_n_clause), 0)] = words_n_clause
                elif self.F.dataset == 'syriac':
                    n_clause_dict[(book, chapter_number, n_gram, 0)] = n_gram
        return n_clause_dict

    def augment_data_with_proper_noun_ids(self, 
                                          n_clause_dict: dict, 
                                          feature: str,
                                          proper_noun_ids_dict: dict,
                                          prob: float,
                                          ):
        """
        Augment the sequences by swapping ids of proper nouns with the ids of other proper nouns.
        """
    
        n_clause_dict_augmented = {}
    
        for key, words in n_clause_dict.items():
            counter = 1 # include a counter to avoid double keys.
            bo, ch, cl, nr = key
            n_clause_dict_augmented[key] = words
            proper_noun_ids = [w for w in words if eval(f'self.F.{feature}.v(w)') in proper_noun_ids_dict.keys()]
            if not proper_noun_ids:
                continue
            for word_idx, word_id in enumerate(words):
                for proper_key, proper_ids in proper_noun_ids_dict.items():
                    random.seed(counter)
                    selected_proper_ids = random.choices(proper_ids, k=int(prob * len(proper_ids)))
                    if eval(f'self.F.{feature}.v(word_id)') == proper_key and self.F.lex.v(word_id) not in {'MRJ>', 'JHWH/'} :
                        for prop_noun_id in selected_proper_ids:
                            words_copy = list(words).copy()
                            words_copy[word_idx] = prop_noun_id
                            n_clause_dict_augmented[(bo, ch, cl, counter)] = tuple(words_copy)
                            counter += 1
        return n_clause_dict_augmented

    def make_proper_noun_dict(self,
                              feature: str,
                              proper_noun_values: list,
                              ):
        """
        Inputs:
            feature: str TF feature which contains the relavant information.
            proper_noun_values:list values that different kinds of proper nouns have.
            F FNode features object of Text-Fabric.
        Output:
            proper_noun_ids_dict: dict key: string from proper_noun_values, value: list with TF ids of unique proper nouns
        """
        proper_noun_ids_dict = {}
        for ls in proper_noun_values: 
            proper_noun_ids_dict[ls] = list({self.F.lex.v(w):w for w in self.F.otype.s('word') if eval(f'self.F.{feature}.v(w)') == ls}.values())
        return proper_noun_ids_dict

class MorphemePreparator:
    def __init__(self, F, alphabet_dict_lat_heb):
        self.F = F
        self.alphabet_dict_lat_heb = alphabet_dict_lat_heb
        
    def make_morph_string(self, morph_list, morph_marker_dict):
        morph_list_with_markers = []
        for idx, morph in enumerate(morph_list):
            if morph:
                morph = morph + morph_marker_dict.get(idx, '')
                morph_list_with_markers.append(morph)
        morph_string_with_markers = ' '.join(morph_list_with_markers)
        return morph_string_with_markers
        
    def convert_ascii_string_to_heb_script(self, ascii_string):
        return ''.join([self.alphabet_dict_lat_heb[char] for char in ascii_string])
        
    def make_morpheme_dict(self, n_clause_dict, word_class, relevant_chars_utf8, morph_marker_dict):
        """
        returns:
        all_morph_strings_with_markers
        keys: (book: str, (clause ids))
        values: hebrew string with morphemes as separate words (with markers) for morpheme types
        """
        all_morph_strings = {}

        for key, words in n_clause_dict.items():
            morphemes_in_clauses = []
    
            for w in words:
                word = word_class(self.F, w, relevant_chars_utf8, self.convert_ascii_string_to_heb_script)
                morph_string = self.make_morph_string(word.morph_list, morph_marker_dict)
                morphemes_in_clauses.append(morph_string)
            
            all_morph_strings[key] = ' '.join(morphemes_in_clauses)

        return all_morph_strings
    
    @staticmethod
    def remove_duplicates(morpheme_dataset):
        swapped_morpheme_dataset = {v:k for k, v in morpheme_dataset.items()}
        morpheme_dataset = {v:k for k, v in swapped_morpheme_dataset.items()}
        return morpheme_dataset
        
    @staticmethod
    def save_data_as_txt_file(file_name, morpheme_dataset):
        with open(file_name, 'w', encoding='utf8') as f:
            for heb_text in morpheme_dataset.values():
                f.write(heb_text + '\n')

def prepare_hebrew_and_syriac_data(augment_hebrew, augment_hebrew_prob, add_syriac, 
                                   augment_syriac, augment_syriac_prob, seq_length, data_file_name,
                                   k_folds,
                                   Fheb, Lheb, Theb,
                                   Fsyr, Lsyr, Tsyr):
    print('Prepare data')
    general_data = GeneralData()
    node_preparator_heb = TFNodePreparator(seq_length, 'clause', Fheb, Lheb, Theb)
    chapter_nodes_folds = node_preparator_heb.split_chapter_list_in_k_equal_lists(k_folds)
    test_chapters = chapter_nodes_folds[0]
    node_preparator_heb.save_test_chapter_ids(test_chapters)
    
    n_words_dict = node_preparator_heb.make_words_n_gram_dict(test_chapters)
    print('Unaugmented_hebrew', len(n_words_dict))

    if augment_hebrew:
        print('Augment Hebrew data')
        hebrew_proper_noun_keys = ['topo', 'pers']
        hebrew_proper_noun_ids_dict = node_preparator_heb.make_proper_noun_dict('nametype', hebrew_proper_noun_keys)
        n_words_dict = node_preparator_heb.augment_data_with_proper_noun_ids(n_words_dict, 
                                                                             'nametype', 
                                                                              hebrew_proper_noun_ids_dict, 
                                                                              augment_hebrew_prob)
        print('Augmented hebrew', len(n_words_dict))
    morpheme_preparator_heb = MorphemePreparator(Fheb, general_data.alphabet_dict_lat_heb)
    morpheme_dataset = morpheme_preparator_heb.make_morpheme_dict(n_words_dict, 
                                                                  HebrewWord, 
                                                                  general_data.relevant_chars_utf8, 
                                                                  general_data.heb_morph_marker_dict)
    print(len(morpheme_dataset))
    
    if add_syriac:
        print('Add Syriac data')
        node_preparator_syr = TFNodePreparator(seq_length*5, 'word', Fsyr, Lsyr, Tsyr)
        syr_n_words_dict = node_preparator_syr.make_words_n_gram_dict()
        print('unaugmented syriac', len(syr_n_words_dict))

        if augment_syriac:
            syriac_proper_noun_keys = ['prop', 'gntl']
            syriac_proper_noun_ids_dict = node_preparator_syr.make_proper_noun_dict('ls', syriac_proper_noun_keys)
            syr_n_words_dict = node_preparator_syr.augment_data_with_proper_noun_ids(syr_n_words_dict, 
                                                                                      'ls', 
                                                                                      syriac_proper_noun_ids_dict, 
                                                                                      augment_syriac_prob)
            print('augmented syriac', len(syr_n_words_dict))
        morpheme_preparator_syr = MorphemePreparator(Fsyr, general_data.alphabet_dict_lat_heb)
        syr_morpheme_dict = morpheme_preparator_syr.make_morpheme_dict(syr_n_words_dict, 
                                                                   SyriacWord, 
                                                                   general_data.relevant_chars_utf8,
                                                                   general_data.syr_morph_marker_dict)
        morpheme_dataset = {**morpheme_dataset, **syr_morpheme_dict}
        print('With Syriac added', len(morpheme_dataset))
        
    morpheme_dataset = morpheme_preparator_heb.remove_duplicates(morpheme_dataset)
    print('After removal of duplicates', len(morpheme_dataset))
    morpheme_preparator_heb.save_data_as_txt_file(data_file_name, morpheme_dataset)
    return morpheme_dataset
    