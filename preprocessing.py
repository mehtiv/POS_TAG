import re
import nltk

def split_sentence(sentence):
    """
    [summary]
        split sentence by '.' and ','
    Arguments:
        sentence {string}
    Returns:
        list[string] 
    """
    
    dot_splited_sentence = [element for element in sentence.split(".") if element != ""]
    return_splited_sentence = [text.rstrip().lstrip() for element in dot_splited_sentence for text in element.split(',') if text != ""]    
    
    return return_splited_sentence

def change_charachters(sentence):
    """
    [summary]
        erase spaces from right and left and delete all the non alpha-numeric carachters
    Arguments:
        sentence {sentence}
    
    Returns:
        string
    """
    return re.sub('[^A-Za-z0-9]+', ' ', sentence).lstrip().rstrip()

def formatting_sentences(sentences):
    """
    [summary]
        clean sentences by applying the split and change carachters functions
    Arguments:
        sentences {list[string]}
    Returns:
        list[string]
    """
    if type(sentences) == str:
        sentences = [sentences]
    
    formatted_sentences = []

    formatting_sentences = [element for sentence in sentences for element in split_sentence(sentence) ]
    formatted_sentences = [change_charachters(element) for element in formatting_sentences]

    return formatted_sentences

def pos_tagging_sentence(sentences):
    """
    [summary]
        tag elements of a sentence (gramatical tags)
    Arguments:
        sentences {list[string]}
    Returns:
        list[list[tuple]] -- tuple('string', 'string') = tuple(word, word's tag)
    """
    if type(sentences) is str:
        sentences = [sentences]
    
    tagged_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged_sentences.append(nltk.pos_tag(tokens))
    
    return tagged_sentences