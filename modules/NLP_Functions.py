#*************************************************************************************
#Generic libs
import os
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import csv
from math import isnan

#**********************************************************************************************************
# string & nlp librairies
import re
import string
# we want to keep the # char and the apostrophe ' lemmatizer will get rid of it
punct = re.sub('[\']', '',string.punctuation)
annotation_list =['bab', 'gue', 'uni', 'rep', 'inq', 'ono', 'hes', 'mis', 'disf']
pos_tags_to_keep = ['ADJ','ADV','AUX','INTJ','NOUN','PRON','ADJ','VERB']

#***************spaCy and its components
import spacy
nlp = spacy.load("en_core_web_sm")
#stopwords = nlp.Defaults.stop_words

#***************nltk and its components
import nltk
nltk.download('punkt')

# The multi-word expression tokenizer is a rule-based
from nltk.tokenize import MWETokenizer, word_tokenize
mwe_tokenizer = MWETokenizer()

from nltk.stem import LancasterStemmer
lancaster=LancasterStemmer()

#***************Pyenchant and its components
import enchant
en_dic = enchant.Dict("en_US")


#***********************************************************************************************************
def is_nan(x):
    return isinstance(x, float) and isnan(x)

def age_to_months(age):
    if not is_nan(age) and (age!=''):
        years_l = age.split(';')
        months_l = age.split('.')[0].split(';')
        age_in_months=0
        if (len(years_l)!=0) and (years_l[0]!=''):
            years = float(years_l[0])
            age_in_months = years *12

        if (len(months_l)!=0) and (months_l[1]!=''):
            months = float(months_l[1])
            age_in_months = age_in_months + months
        return age_in_months
    else:
        return age
#**********************************************Common STRING Functions**************************************
#***********************************************************************************************************
#***********************************************************************************************************

def list_to_txt(txt_file_path, list_):
    '''
    This function saves the content of a list in a text file
    '''
    textfile = open(txt_file_path+".txt", "w")
    for i, e in enumerate(list_):
        if i != len(list_)-1:
            textfile.write(e + "\n")
        else:
            textfile.write(e)
    textfile.close()

def len_speech(speech):
    '''
    This function computes the length of a given text
    '''
    return len(str(speech))


def search_pattern(pattern, string):
    '''
    This function finds all the occurrences of a given pattern in a string
    '''
    prog = re.compile(pattern)
    return prog.findall(string)


def search_all_rows(df, column, pattern, txt_path, distinct=1):
    '''
    This function finds all the occurrences of a given pattern in a dataset
    '''
    not_list =[]
    for index, row in df.iterrows():
        not_list.extend(search_pattern(pattern, row[column]))

    if distinct == 1:
        list_to_txt(txt_path, set(not_list))
        return set(not_list)
    else:
        list_to_txt(txt_path, not_list)
        return not_list


def count_all_rows(df, column, pattern):
    '''
    This function counts the occurrences of a given pattern in a dataset
    '''
    not_list =[]
    for index, row in df.iterrows():
        not_list.extend(search_pattern(pattern, row[column]))

    return len(not_list)


def isEnglish(string):
    '''
    This function checks whether a given string contains non ascii characters
    '''
    try:
        string.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def isEnglish_all_rows(df, column, txt_path):
    '''
    This function counts the occurrences of strings containing non ascii characters
    It saves the found strings in a text file
    '''
    list_latin_speech =[]
    for index, row in df.iterrows():
        if not isEnglish(row[column]):
            list_latin_speech.append(row[column])

    list_to_txt(txt_path, set(list_latin_speech))
    return len(list_latin_speech)


def punctuation(text):
    pattern = '['+ punct+']'
    return set(re.findall(pattern, text))


def punctuation_all_rows(df, column):
    punct_set=set([])
    for index, row in df.iterrows():
        punct_set = set.union(punct_set, punctuation(row[column]))
    return punct_set

#***********************************************************************************************************
#**********************************************NLP Functions*******************************************
#***********************************************************************************************************
#***********************************************************************************************************

def common_preprocess(sentence):
    clean_sentence = ''
    if not is_nan(sentence):
        # remove numbers
        sentence_w_num = ''.join(i for i in sentence if not i.isdigit())
        # lower and remove punctuation 
        sentence_w_punct = "".join([i.lower() for i in sentence_w_num if i not in punct])
        #Remove extra spaces, tabs, and line breaks
        clean_sentence= " ".join(sentence_w_punct.split())

    return clean_sentence


def lemmatizer(sentence):
    doc = nlp(sentence)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return lemmatized_sentence


def tokenizer(sentence):
    tokenized_sentence = mwe_tokenizer.tokenize(word_tokenize(sentence))
    return tokenized_sentence


def extract_meaningful_words(sentence):
    tokenized_sentence = tokenizer(sentence)
    english_sentence = ' '.join([token for token in tokenized_sentence if token in annotation_list or en_dic.check(token)])
    return english_sentence


def stemmer(sentence):
    tokenized_sentence = tokenizer(sentence)
    stemmed_sentence = ' '.join([lancaster.stem(token) if token not in annotation_list else token for token in tokenized_sentence])
    return stemmed_sentence


def structure(sentence):
    doc = nlp(sentence)
    structured_sentence = " ".join([token.lemma_ for token in doc if (token.pos_ in pos_tags_to_keep) or (token.lemma_ in annotation_list)])
    return structured_sentence

#***********************************************************************************************************
#**********************************************Preprocessing Functions**************************************
#***********************************************************************************************************
#***********************************************************************************************************

def preprocess(data, column, preprocessed_dataset_path): 
    '''
    column : raw text
    create new columns:
        'clean_annotated_speech': clean 'speech'
            1) annotate/label the speech 
            2) remove foreign language speech, Mojibake, numbers, punctuation and extras spaces, tabs and line breaks   
        'lemmatized_speech': lemmatize 'clean_annotated_speech'
        'meaningful_speech': meaningful 'lemmatized_speech'
            extract only meaningful english words (e.g., bibobi is not a meaningful word)
        'structured_speech': structure 'meaningful_speech'
            keep only some forms of words {subject, noun, verb, adj, adv}
        'stemmed_lemmatized_speech':  stemm 'lemmatized_speech'        
        'stemmed_meaningful_speech':  stemm 'meaningful_speech'  (shorter sentences)      
    '''

    df = data.copy(deep=True)
    df['clean_annotated_speech'] = df[column]

    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    step = 1
    print(f'Step {step}: Annotate Speech')
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        speech = row['clean_annotated_speech']

        # step I: handle non ascii and Mojibake (e.g., question mark in a square)
        #print(f'Step {step}: Non ASCII and Mojibake')
        speech = speech.encode("ascii", "ignore").decode()
        speech = ''.join(filter(lambda x:x in string.printable, speech))
        # *******************************************************************************************
        # step II: annotate speech
        # we choose to annotate the text using #short_Term because # is not used in the dataset
        #in such a way, we will not confound neither the # nor the used word with other used characters and words and 
        # we will not bias the training after that
        #eg., we used the term 'INQ' instead of 'INC' to annotate incomplete speech to not confound the annotation with the word INC(Incorporation)
        # and we used 'DISF' to annotate disfluency because many children spell 'this' as 'dis'
        # e.g., CHI: yes I am babbling : there is no babbling and the child expresses clearly what he is doing
        #       CHI: uh bbabibou[:babbles] what : here the child is babbling because he can't clearly and easily express 

        # *******************************************************************************************
        #print(f'Step {step}: Babbling')
        #Babbling #bab
        # @b, @u, @wp, list([... babble, babbling..]), list(&=babbl)
        patterns = ['\[[^\]]*babbl[^\]]*\]', '&=babbl[\w]*', '@b','@u','@wp']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    speech = speech.replace(pattern, ' BAB ')   

        # *******************************************************************************************
        #print(f'Step {step}: Guess')
        #best Guess #gue
        # pattern : [?], [=? ...]
        patterns = ['\[\?\]', '\[\=\?[^\]]*\]']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    speech = speech.replace(e, ' GUE ')

        # *******************************************************************************************
        #print(f'Step {step}: Unintelligible')
        # Unintelligible #uni
        # patterns: ['[=jargon]', 'xxx', 'xxxx', 'yyy']
        patterns = ['[=jargon]','xxxx', 'xxx', 'yyy']
        for pattern in patterns:
            if pattern in speech:
                speech = speech.replace(pattern, ' UNI ')

        # *******************************************************************************************        
        #print(f'Step {step}: Repetition')         
        #repetition #rep
        # patterns: ['[=repeat..]', [x n], [/]]
        patterns = ['\[[^\]]*repeat[^\]]*\]', '\[x [\d^\]]*\]', '\[[\/^\]]*\]']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    N = re.search(r'\d+', e)
                    annotation = ''
                    if N:
                        for i in range(int(N.group())):
                            annotation = annotation + ' REP '
                    else:
                        annotation = ' REP '
                    speech = speech.replace(e, annotation)

        # *******************************************************************************************
        #print(f'Step {step}: Incompletion')         
        #incompletion #inQ
        # patterns: [+..., +..?]
        patterns = ['+...', '+..?']
        for pattern in patterns:
            if pattern in speech:
                speech = speech.replace(pattern, ' INQ ')

        # *******************************************************************************************
        #print(f'Step {step}: Onomatopoeia')         
        #onomatopoeia #ono
        # patterns: @o
        pattern = '@o'
        if pattern in speech:
            speech = speech.replace(pattern, ' ONO ')

        # *******************************************************************************************      
        #print(f'Step {step}: Hesitation')    
        #hesitation #hes
        # patterns: [[//], [///], &+, [/?] ]
        patterns = ['[//]','[///]', '&+', '[/?]']
        for pattern in patterns:
            if pattern in speech:
                speech = speech.replace(pattern, ' HES ')

        # *******************************************************************************************
        #print(f'Step {step}: Misspeling')    
        #misspeling #misspell
        # patterns: [: text]
        pattern = '\[\:[^\]]*\]'
        pat = search_pattern(pattern, speech)
        if len(pat)!=0:
            for e in set(pat):
                speech = speech.replace(e, ' MIS ')

        # *******************************************************************************************
        #print(f'Step {step}: Events')       
        #delete events &=
        #generally events are visual notes of the transcripter on some actions that the child is doing
        # we are only intereseted in text( what did the child say not what did he do or how did he act)
        pattern = '&=[\w\:\_]*'
        pat = search_pattern(pattern, speech)
        if len(pat)!=0:
            for e in set(pat):
                speech = speech.replace(e, '')

        # *******************************************************************************************
        #print(f'Step {step}: Disfluency')          
        #handle patterns of disfluency #disfluency
        patterns = ['\(.\)', '\(..\)', '\(...\)', '\[/-\]', '&[\w]*:[\w]*','&-[\w]*','&[\w]*']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    speech = speech.replace(e, ' DISF ')

        #handle other patterns of disfluency ['[\w]*:[\w]*', '[\w]*\^[\w]*']
        patterns = ['[\w]*:[\w]*', '[\w]*\^[\w]*']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    if pattern == '[\w]*:[\w]*':
                        speech= speech.replace(e, e.replace(':',''))
                    else:
                        speech = speech.replace(e, e.replace('^',''))

                    speech = speech + ' DISF '

        # *******************************************************************************************
        #step III: delete useless notations
        # we coonsider a notation as useless if it is not relevant to the problem we are dealing with
        # example: www: untranscribted material (the child is looking at pictures)

        #print(f'Step {step}: Useless Annotations')    
        patterns = ['\[[^\]]*\]', '\swww\s', '[\t]', '[\>]','[\<]','[\)]', '[\(]']
        for pattern in patterns:
            pat = search_pattern(pattern, speech)
            if len(pat)!=0:
                for e in set(pat):
                    speech = speech.replace(e, ' ')

        # *******************************************************************************************           
        # save modifications
        df.at[index, 'clean_annotated_speech'] = speech 
    #end for
    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    step+=1
    print(f'Step {step}: Common Preprocessing')  
    # apply some other common preprocessing          
    df['clean_annotated_speech'] = df['clean_annotated_speech'].progress_apply(common_preprocess)

    # *******************************************************************************************
    #Step IV: add new columns
    step+=1
    print(f'Step {step}: Lemmatization')        
    df['lemmatized_speech'] = df['clean_annotated_speech'].progress_apply(lemmatizer)

    # *******************************************************************************************
    step+=1
    print(f'Step {step}: Extract Meaningful Speech') 
    df['meaningful_speech'] = df['lemmatized_speech'].progress_apply(extract_meaningful_words)

    # *******************************************************************************************
    step+=1
    print(f'Step {step}: Structure Speech') 
    df['structured_speech'] = df['meaningful_speech'].progress_apply(structure)

    # *******************************************************************************************
    step+=1
    print(f'Step {step}: Stem Lemmatized Speech') 
    df['stemmed_lemmatized_speech'] = df['lemmatized_speech'].progress_apply(stemmer)    

    # *******************************************************************************************
    step+=1
    print(f'Step {step}: Stem Structured Speech') 
    df['stemmed_structured_speech'] = df['structured_speech'].progress_apply(stemmer)      

    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    # *******************************************************************************************
    print('Ouf!! Save ...')
    # save the preprocessed data in a csv file
    if os.path.exists(preprocessed_dataset_path):
        os.remove(preprocessed_dataset_path)
    df.to_csv(preprocessed_dataset_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f'Preprocessing is done, you find your clean data at {preprocessed_dataset_path}')
   
    
#***********************************************************************************************************
#**********************************************Features Engineering Functions*******************************************
#***********************************************************************************************************
#***********************************************************************************************************        
        
def annotation_count(sentence, annotations_dic):
    tokenized_sentence = tokenizer(sentence)
    for token in tokenized_sentence:
        if token in annotations_dic.keys():
            annotations_dic[token] +=1
    return annotations_dic

def count(sentence):
    tokenized_sentence = tokenizer(sentence)
    tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
    return len(tokenized_sentence)
 
def count_diff_words(sentence):
    count = 0
    if not is_nan(sentence):
        tokenized_sentence = tokenizer(sentence)
        tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
        count = len(set(tokenized_sentence))
    return count      

def density(sentence):
    den = 0
    if not is_nan(sentence):
        tokenized_sentence = tokenizer(sentence)
        tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
        sentence_to_str = "".join([token for token in tokenized_sentence])
        den = len(sentence_to_str)  
    return den    

def remove_stopwords(sentence):
    if not is_nan(sentence):
        tokenized_sentence = tokenizer(sentence)
        tokenized_sentence = [token for token in tokenized_sentence if token not in stopwords]
        sentence = " ".join([token for token in tokenized_sentence])
    return sentence

# ***************************************************************************************************************
def extract_features(dataset, csv_path):
    data = dataset.copy(deep=True)
    
    #convert sex to integer (Male =1; female =0)
    data.sex = data.sex.apply(lambda x: 1 if x=='male' else 0)
    
    #1. compute the length of each type of speech
    print('Step 1: Compute speech length 1/3')
    data['len_clean_annotated_speech'] = data['clean_annotated_speech'].astype(str).apply(count)
    print('Step 1: Compute speech length 2/3')
    data['len_meaningful_speech'] = data['meaningful_speech'].astype(str).apply(count)
    print('Step 1: Compute speech length 3/3')
    data['len_structured_speech'] = data['structured_speech'].astype(str).apply(count)

    #2. compute number of each annotation in the speech
    print('Step 2: Compute number of each annotation')
    #2.1 create columns for the number of each annotation in a given speech
    data['n_bab']=0
    data['n_gue']=0
    data['n_uni']=0
    data['n_rep']=0
    data['n_inq']=0
    data['n_ono']=0
    data['n_hes']=0
    data['n_mis']=0
    data['n_disf']=0

    #2.2 compute
    for index, row in data.iterrows():
        annotations_dic = {'bab': 0, 'gue': 0, 'uni': 0, 'rep': 0, 'inq': 0, 'ono': 0, 'hes': 0, 'mis': 0, 'disf': 0}
        speech = data.at[index, 'clean_annotated_speech']
        annotations_dic = annotation_count(str(speech), annotations_dic)
        data.at[index, 'n_bab']=annotations_dic['bab']
        data.at[index, 'n_gue']=annotations_dic['gue']
        data.at[index, 'n_uni']=annotations_dic['uni']
        data.at[index, 'n_rep']=annotations_dic['rep']
        data.at[index, 'n_inq']=annotations_dic['inq']
        data.at[index, 'n_ono']=annotations_dic['ono']
        data.at[index, 'n_hes']=annotations_dic['hes']
        data.at[index, 'n_mis']=annotations_dic['mis']
        data.at[index, 'n_disf']=annotations_dic['disf']
        
    print('Step 3: convert age to months')
    #3. compute age in months
    data['age_in_months']= data.age.apply(age_to_months)
    
    print('Step 4: compute number of different used words')
    #3. compute number of different used words
    data['n_diff_words']= data.clean_annotated_speech.apply(count_diff_words)
    
    print('Step 5: compute density of speech')
    #density = nb of characters used in the speech without considering whitespaces and annotations
    data['density']= data.clean_annotated_speech.apply(density)
    
    # save the new dataset
    print('Save!! ...')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    data.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f'Features Extracting is done, you find your extracted data at {csv_path}')
    
# *******************************************************************************************************************************

