## mcu wikidata 
import wikipediaapi

wiki = wikipediaapi.Wikipedia('en')

mcu = "Marvel Cinematic Universe"
page_py = wiki.page(mcu)
# print(page_py.exists())

# print("Page - Summary: %s" % page_py.summary[:100])

wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)
p_wiki = wiki.page(mcu)

mcu_text = p_wiki.text[:2509+len('commercials.')]
# print(mcu_text)


## step 1. coreference resolution
import spacy
import neuralcoref
from spacy.tokenizer import Tokenizer


nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

def coref_resolution(text):
    '''
    주어진 텍스트에 대해 coreference resolution 수행하고
    같은 entity는 같은 표현으로 바꿔주는 함수
    '''
    doc = nlp(text)
    return doc._.has_coref, doc._.coref_clusters, doc._.coref_resolved

doc = nlp(mcu_text)
sentences = [sent.string.strip() for sent in doc.sents]

for s in sentences:
    print(coref_resolution(s))

import re
mcu_text = re.sub('\n',' ', mcu_text)
mcu_text