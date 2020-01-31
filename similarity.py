from preprocessing import *
from gensim.models.wrappers import FastText
import pandas as pd
import numpy as np

NUMBER_DOC = 4124

print("starting loading fasttext")
model = FastText.load_fasttext_format('cc.en.300.bin')
print("fasttext loaded")

def building_idf(path, number_doc):
    data = pd.read_csv(path, sep= " ")
    data.columns = ['frequency', 'word', 'pos', 'num_doc_occ']

    data['idf'] = data.apply(lambda element:np.log(NUMBER_DOC/element['num_doc_occ']), axis=1)

    return data


def sim(element_1, element_2):
    return model.similarity(element_1, element_2)


def simple_similarity_calcul(sentence_1, sentence_2, data): # No filter

    cleared_sentence_1_2 = formatting_sentences([sentence_1, sentence_2])
    cleared_sentence_1, cleared_sentence_2 = cleared_sentence_1_2[0], cleared_sentence_1_2[1]

    tagged_sentence_1 = pos_tagging_sentence(cleared_sentence_1)[0]
    tagged_sentence_2 = pos_tagging_sentence(cleared_sentence_2)[0]

    similarity = 0
    idf_sum = 0
    for element_1 in tagged_sentence_1:
        max_sim_element_1 = 0
    
        for element_2 in tagged_sentence_2:
            try:
                idf_element = data.iloc[data[data['word']==element_1[0].lower()]['idf'].idxmax()].idf
                max_sim_element_1 = max(max_sim_element_1,sim(element_1[0], element_2[0]))
            except:
                idf_element = 0
                max_sim_element_1 = 0
        
        similarity += max_sim_element_1 * idf_element
        idf_sum += idf_element
  
    return similarity/idf_sum



def filtred_simple_similarity_calcul(sentence_1, sentence_2, data): # Filtered

    cleared_sentence_1_2 = formatting_sentences([sentence_1, sentence_2])
    cleared_sentence_1, cleared_sentence_2 = cleared_sentence_1_2[0], cleared_sentence_1_2[1]

    tagged_sentence_1 = pos_tagging_sentence(cleared_sentence_1)[0]
    tagged_sentence_2 = pos_tagging_sentence(cleared_sentence_2)[0]

    similarity = 0
    for element_1, tag_1 in tagged_sentence_1:
        max_sim_element_1 = 0
        for element_2, tag_2 in tagged_sentence_2:
            #if tag_1 == tag_2:
            max_sim_element_1 = max(max_sim_element_1,sim(element_1, element_2))
        
        similarity += max_sim_element_1 
    
    return similarity/len(tagged_sentence_1)



sentence_1 = "As part of Microsoft Accenture Avanade team  we took the concept of RPA a step further Where we incorporate intelligence by deploying cognitive technologies from Microsoft Azure services to RPA platforms  to combine Machine Learning  Natural language Processing and process automation to perform complex tasks without human interference Deep learning in Accenture Insurance Claim Handling using Deep learning (CNN)"
sentence_2 = "Client facing Analysis of the projects needs Estimates of the development charge Specifications writing  Reading Room is a digital agency based in London specialized in the realisation of web services for major international clients It is now part of the IDOX Group I worked on the realisation of web applications and services for major brands and institutions such as UK Government"
sentence_3 = "Self employed regularly taking jobs in web development for individuals or professionals I learned to work with many different web technologies to satisfy my clients by building strong and modern solutions"


data = building_idf('./all.al', NUMBER_DOC)
'''
print("-_-_-_-_-_-_ No filter similarity -_-_-_-_-_-_")
print((simple_similarity_calcul(sentence_1, sentence_2, data) + simple_similarity_calcul(sentence_2, sentence_1, data))/2)
print((simple_similarity_calcul(sentence_1, sentence_3, data) + simple_similarity_calcul(sentence_3, sentence_1, data))/2)
'''
print("-_-_-_-_-_-_ filtered similarity -_-_-_-_-_-_")
print((filtred_simple_similarity_calcul(sentence_1.lower(), sentence_2.lower(), data) + filtred_simple_similarity_calcul(sentence_2.lower(), sentence_1.lower(), data))/2)
print((filtred_simple_similarity_calcul(sentence_1.lower(), sentence_3.lower(), data) + filtred_simple_similarity_calcul(sentence_3.lower(), sentence_1.lower(), data))/2)
