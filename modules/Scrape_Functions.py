# useful libraries
import csv
import re

import requests
from selectorlib import Extractor

from tqdm import tqdm
from time import sleep

# ***************************************************************************************************************************************************************
# scraping functions
def scrape(url, yml_file): 
    '''
    This function is a very basic scraping function that takes:
        an 'url' ready to be scraped
        a yaml file that contains a structured description of what we want to scrape
    it uses two libraries: selectorlib and requests
    '''
    e = Extractor.from_yaml_file(yml_file)
    r = requests.get(url)
    data = e.extract(r.text, base_url=url)
    sleep(1)
    
    return data


def scrape_repositories_urls(yml_file_path, urls_file_path):
    '''
    This function scrapes the urls of the repositories where the cha files (files that contain the conversations) 
    are saved.
    It uses 2 params:
        yml_file_path: the path of the yaml file used to scrape data
        urls_file_path: the path of the text file where we have saved the base urls to be scraped
            these urls are:
                simple (S), i.e., they contain the cha files, so we will be able to immediately scrape cha files  
                multiple (M), i.e., they contain other repositories of cha files, in this case, we should first
                scrape the sub-repositories urls in order to be able to scrape cha files
                
    It uses the very basic predefined scraping function scrape(url, yml_file) and the re(regular expressions) library
    '''

    rep_urls =[]
    with open(urls_file_path, "r") as file:
        for line in file:
            line = line.strip()
            line_l = line.split(',')
            if line_l[1] == 'S':
                rep_urls.append(line_l[0])
            else:
                # 1. scrape repositories urls
                data = scrape(line_l[0], yml_file_path) 

                # 2. filter urls
                if data['names']:
                    for url in data['names']['url']:
                        if re.search(line_l[0], url) is not None and\
                        re.search('cdc', url) is None and\
                        re.search('txt', url) is None and\
                        re.search('=', url) is None:
                            rep_urls.append(url)
    return rep_urls


def scrape_urls_files(yml_file_path, rep_urls_list):
    '''
    This function scrapes the urls of the cha files where the children conversations are saved
    It uses 2 params:
        yml_file_path: the path of the yaml file used to scrape data
        rep_urls_list: the list of repositories urls where the cha files are saved
    It uses very basic predefined scraping function scrape(url, yml_file) and the re(regular expressions) library
    '''
    cha_files_urls =[]
    for url in rep_urls_list:
        data = scrape(url, yml_file_path) 

        #2. filter and save cha files urls
        if data['names']:
            for url in data['names']['url']:
                if re.search('cha', url) is not None:
                        cha_files_urls.append(url)

    return cha_files_urls


def scrape_and_parse_cha_files(cha_files_urls, csv_file_path, asd):
    '''
    This function:
        1) scrapes the cha files where the children conversations are saved
        2) parses the obtained data and feed a csv file with the parsed data
    It uses 2 params:
        cha_files_urls: the urls of the cha files
        csv_file_path: the path of the csv file to be fed with the scraped and parsed data
    It uses 3 libraries: csv, requests, and re
    '''
    with open(csv_file_path,'w') as f:
        fieldnames = [
            "name",
            "age",
            "sex",
            "speech",
            "ASD"
        ]

        writer = csv.DictWriter(f, fieldnames = fieldnames, quoting = csv.QUOTE_NONNUMERIC)
        writer.writeheader()

        # scrape cha files 
        for i, url in tqdm(enumerate(cha_files_urls), total=len(cha_files_urls)):
            r = requests.get(url)
            doc = r.text.split('\n')

            # parse the obtained data 
            data_dic = {'name':'', 'age':'', 'sex':'', 'ASD': asd, 'speech':[]}
            for j, line in tqdm(enumerate(doc), total=len(doc)):
                # generic info
                if re.search('ID:', line) is not None and re.search('CHI', line) is not None:
                    s = line.split('|')
                    data_dic['name'] = s[1]
                    data_dic['age'] = s[3]
                    data_dic['sex'] = s[4]
                # speech
                if re.search('\*CHI', line) is not None:
                    s = line.split(':',1)
                    data_dic['speech'] = s[1]
                    writer.writerow(data_dic)

            sleep(5)
    print('END')


# ******************************************************************************************************************************************
# other useful helpers
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
    
def abs_age(age):
        '''
        extract absolute age
        '''
        if age==age:
            return int(age.split(';')[0])
        else:
            return age