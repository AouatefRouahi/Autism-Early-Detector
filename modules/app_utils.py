import pandas as pd
import os

from datetime import datetime, date

import nltk
nltk.download('punkt')
from nltk.tokenize import MWETokenizer, word_tokenize
mwe_tokenizer = MWETokenizer()
import spacy
nlp = spacy.load("en_core_web_sm")
import enchant
en_dic = enchant.Dict("en_US")

#******************************************************************Image Functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_data_content():
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

def csv_register(name, birth_date, sex):
    dict_child = {"name" : [name], "birth_date" : [birth_date], "sex" : [sex]}
    df_child = pd.DataFrame.from_dict(dict_child)
    df_child.to_csv("data/data_child.csv", index=False)

def speech_register(name, birth_date, sex, speech):
    dict_child = {"name" : [name], "birth_date" : [birth_date], "sex" : [sex], "speech":[speech]}
    df_child = pd.DataFrame.from_dict(dict_child)
    df_child.to_csv("data/data_speech_child.csv", index=False)

#******************************************************************Text Functions

def age_months(age):
    d = datetime.fromisoformat(age)
    today = date.today()
    num_months = abs((today.year - d.year) * 12 + (today.month - d.month))
    return num_months

def count(sentence):
    tokenized_sentence = tokenizer(sentence)
    tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
    return len(tokenized_sentence)

def tokenizer(sentence):
    #print(mwe_tokenizer.tokenize(word_tokenize(sentence)))
    tokenized_sentence = mwe_tokenizer.tokenize(word_tokenize(sentence))
    return tokenized_sentence

def tokenizer2(sentence):
    tokenized_sentence = sentence.split(" ")
    tokenized_sentence = list(filter(bool, tokenized_sentence))
    print(tokenized_sentence)
    return tokenized_sentence

def annotation_count(sentence, annotations_dic):
    tokenized_sentence = tokenizer(sentence)
    for token in tokenized_sentence:
        if token in annotations_dic.keys():
            annotations_dic[token] +=1
    return annotations_dic

def count_diff_words(sentence):
    tokenized_sentence = tokenizer(sentence)
    tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
    count = len(set(tokenized_sentence))
    return count      
 
def density(sentence):
    tokenized_sentence = tokenizer(sentence)
    tokenized_sentence = [token for token in tokenized_sentence if token not in annotation_list]
    sentence_to_str = "".join([token for token in tokenized_sentence])
    den = len(sentence_to_str)  
    return den    

def extract_meaningful_words(sentence):
    tokenized_sentence = tokenizer(sentence)
    english_sentence = ' '.join([token for token in tokenized_sentence if token in annotation_list or en_dic.check(token)])
    return english_sentence

def structure(sentence):
    doc = nlp(sentence)
    structured_sentence = " ".join([token.lemma_ for token in doc if (token.pos_ in pos_tags_to_keep) or (token.lemma_ in annotation_list)])
    return structured_sentence

def replace(sentence, dic):
    subs = {v:k for k,v in dic.items()}
    tokenized_sentence = tokenizer(sentence)
    s = [subs.get(item,item)  for item in tokenized_sentence]
    return ' '.join([e for e in s])

def extract_features(data, annotations_dic):
    data['speech']= data['speech'].apply(replace, args=([dic]))
    data['len_speech']=data['speech'].apply(count)
    data['meaningful_speech']=data['speech'].apply(extract_meaningful_words)
    
    data['structured_speech']=data['speech'].apply(structure)
    data['len_meaningful_speech']=data['meaningful_speech'].apply(count)
    data['len_structured_speech']=data['structured_speech'].apply(count)

    data['n_bab']=0
    data['n_gue']=0
    data['n_uni']=0
    data['n_rep']=0
    data['n_inq']=0
    data['n_ono']=0
    data['n_hes']=0
    data['n_mis']=0
    data['n_disf']=0
    for index, row in data.iterrows():
        speech = data.at[index, 'speech']
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
        
    data['age_months']=data['birth_date'].apply(age_months)

    data['n_diff_words']= data['speech'].apply(count_diff_words)
    data['density']= data['speech'].apply(density)
    data.sex = data.sex.apply(lambda x: 1 if x=='male' else 0)
    
    return data