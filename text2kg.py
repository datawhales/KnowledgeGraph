import spacy
import neuralcoref
import opennre
from itertools import permutations
from nltk.tokenize import sent_tokenize
from pandas import DataFrame

def coreference_resolution(text):
    '''
    주어진 텍스트에 대해 coreference resolution을 수행하고
    그 결과에 맞게 대표 단어들로 대체시켜 다시 text를 반환하는 함수
    '''
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(text)
    return doc._.coref_resolved

def NER(text):
    '''
    주어진 text에 대해 named entity recognition을 수행하고
    (entity 이름, (entity의 시작 인덱스, entity의 끝 인덱스), entity 종류)의 튜플들이
    담긴 리스트를 반환하는 함수
    '''
    nlp = spacy.load('en')
    doc = nlp(text)
    ent_lst = [(ent.text, (ent.start_char, ent.end_char), ent.label_) for ent in doc.ents]
    
    return ent_lst

def RE(text, head_entity_idx, tail_entity_idx, RE_model):
    '''
    주어진 text에 대해 head entity와 tail entity의 위치를 알고 있을 때
    relation extraction을 수행하고 entity 간의 관계를 반환하는 함수
    '''
    model = opennre.get_model(RE_model)
    result = model.infer({'text': text, 'h':{'pos': head_entity_idx}, 't': {'pos': tail_entity_idx}})
    
    return result

def txt2triple(text, RE_model, relation_threshold=0.8):
    '''
    주어진 텍스트에 대해 coreference resolution을 수행하고
    대체시킨 text에서 named entity recognition을 통해 찾은
    entities에 대해 relation을 추출하는 함수
    '''
    # coreference resolution
    text = coreference_resolution(text)
    
    sentences = [x for x in sent_tokenize(text)]
#     print(sentences)
    triple_list = []
    
    for sent in sentences:
        # named entity recognition
        ent_info = NER(sent)
#         print(ent_info)
        if len(ent_info) >= 2:
            candidates = list(permutations(ent_info, 2))
    
            # relation extraction
            for cand in candidates:
                h, t = cand
            
                h_text, t_text = h[0], t[0]
                h_idx, t_idx = h[1], t[1]
                h_label_, t_label_ = h[2], t[2]
                
                r = RE(text, h_idx, t_idx, RE_model)
                r_text, r_prob = r[0], r[1]
                
                if h_text != t_text and r_prob > relation_threshold:
                    triple = (sent, (h_text, r_text, t_text), r_prob)
                    triple_list.append(triple)
                    
    df = DataFrame(triple_list, columns=['original text','triple','confidence level'])
    df.sort_values(by=['confidence level'], ascending=False, inplace=True)
    
#     return triple_list if triple_list else None
    return df if not df.empty else None