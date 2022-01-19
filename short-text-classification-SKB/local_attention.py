#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import tqdm
import spacy
dataset = "ohsumed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replaceWord2Sememe(embedding_dict, skb_dict, docs_tuple):
    '''
        input: 
        process: if the word of sentence in skb && else if the freqency of word less then threshold:
                    replace the word to sememe:
                        if the sense of word only once:
                            straightforward replace else more thinking... of (sense dismatching- now leave aside)
                        else:
                            search the sentence embedding of docs by look-up embedding dictionary
                            for building the word embedding with weighted sum of sentence:
                                senses cosin = list
                                for index, sense the enumerate(senses):
                                    word cosin = list
                                    for sememe in sense:
                                        compare both that the embedding of the word and the sememe of the sense
                                        append the cosin value to word cosin list
                                    keep minimum of senses cosin to append the senses cosin list
                                get the index of minimum value for sense senses list
                                get the word via index with this word senses of SKB-DA
                            replace  
        return: docs list replaced with sememe
    '''
    sememe_docs_list = []

    '''
        threshold = np.mean(list(docs_tuple[0].values())) + np.std(list(word_freq.values())) /\
                np.mean(list(word_freq.values()))
    '''
    threshold = 1000
    for sentence in tqdm.tqdm(docs_tuple[2]):
        sentence_replace = []
        for word in sentence.split():
            if word in skb_dict.keys() and docs_tuple[0][word] < threshold:
                print(skb_dict)
                for skb_word, senses in skb_dict:
                    if len(senses) == 1:
                        sentence_replace += list(sense[0][1])
                    else:
                        senses_cos_list = []
                        for (_, sememe_set) in senses:
                            senses_cos_list.append(max(localAttention(\
                                                sentence, word, list(sememe_set), embedding_dict)))
                        senses_cos_list_max_index = senses_cos_list.index(max(senses_cos_list))
                        sentence_replace += list(senses[senses_cos_list_max_index][1])          
            else:
                sentence_replace.append(word)
                
        sememe_docs_list.append(" ".join(sentence_replace))
    return sememe_docs_list

def removeWikiSenseOnSKBDA(skb_da_dict):
    '''uniform for SKB-DA sense'''
    skb_da_pure_dict = {}
    skb_da_cdv_set = set()
    for word, sense in tqdm.tqdm(skb_da_dict.items()):
        for (pos, sememe_set) in sense:
            if " (" in word:
                if len(word.split(" (")) == 3:
                    word, sense1,sense2 = word.split(" (")
                    sense1 = sense1.replace(")","")
                    sense2 = sense2.replace(")","")
                    if word not in skb_da_pure_dict.keys():
                        skb_da_pure_dict[word] = []
                    #sememe_set.add(sense1)
                    #sememe_set.add(sense2)
                    sememe_set.discard(word)
                    skb_da_pure_dict[word].append((sense1+" - "+sense2,sememe_set))
                    skb_da_cdv_set =  skb_da_cdv_set | sememe_set
                    continue
                word, sense = word.split(" (")
                sense = sense.replace(")","")
                if word not in skb_da_pure_dict.keys():
                    skb_da_pure_dict[word] = []
                #sememe_set.add(sense)
                sememe_set.discard(word)
                skb_da_pure_dict[word].append((sense,sememe_set))
                skb_da_cdv_set =  skb_da_cdv_set | sememe_set
            else:
                if word not in skb_da_pure_dict.keys():
                    skb_da_pure_dict[word] = []
                sememe_set.discard(word)
                skb_da_pure_dict[word].append((pos,sememe_set))
                skb_da_cdv_set =  skb_da_cdv_set | sememe_set
    print("#all lexicon of SKB-DA: {}; #CDV of SKB-DA: {}".format(len(skb_da_pure_dict),len(skb_da_cdv_set)))
    return skb_da_pure_dict,skb_da_cdv_set

def loadGloveModel(gloveFile):
    '''Loading Glove Model'''
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in tqdm.tqdm(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


doc_content_tuple: tuple = np.load("../data/corpus/" + dataset+".clean.npy", allow_pickle=True).tolist()
skb_da = np.load("../sememe_dataset/skb_ad_dict.npy", allow_pickle=True).tolist()
skb_da_pure,skb_da_pure_cdv_set = removeWikiSenseOnSKBDA(skb_da)
glove_840B_300d_common_crawl = loadGloveModel("../data/embeddings/glove.840B.300d.common_crawl.txt")

sememe_docs_list = replaceWord2Sememe(glove_840B_300d_common_crawl,\
                                      skb_da_pure,\
                                      doc_content_tuple)

np.save("sememe_docs_list_with_local_attention",sememe_docs_list)
