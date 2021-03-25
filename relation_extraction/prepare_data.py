import json
import random
import os
import sys
import pdb
import re
import torch
import argparse
import numpy as np
from tqdm import trange
from collections import Counter, defaultdict

def filter_sentence(sentence):
    '''문장 필터링 함수
    
    Filter sentence:
        - head mention과 tail mention이 같은 문장
        - head mention과 tail mention이 overlap 되는 경우
        
    Args:
        문장은 python dictionary 형태.
        문장이 tokenize된 형태로 주어지고, head entity, tail entity, relation에 대한 정보가 주어짐
        entity의 경우 'pos'는 mention의 인덱스,'name'은 mention span, 'id'는 wikidata id를 의미
        relation의 경우 wikidata id만 주어짐
        
        문장 예시:
        {
            'tokens': ['Microsoft','was','founded','by','Bill','Gates','.'],
            'h': {'pos': [[0]], 'name': 'Microsoft', 'id': Q123456},
            't': {'pos': [[4,5]], 'name': 'Bill Gates', 'id': Q2333},
            'r': 'P1'
        }
    Returns:
        True 또는 False를 반환. filtering해야 하는 문장이 들어오면 True를 반환. 그 외에는 False
        
    Raises:
        문장 형태가 위의 예시와 다르면 key not found error를 발생시킴
    '''
    head_pos = sentence["h"]["pos"][0]   # head_pos = [0]
    tail_pos = sentence["t"]["pos"][0]   # tail_pos = [4, 5]
    
    if sentence["h"]["name"] == sentence["t"]["name"]: return True
    
    if head_pos[0] >= tail_pos[0] and head_pos[0] <= tail_pos[-1]: return True
    
    if head_pos[0] <= tail_pos[0] and head_pos[-1] >= tail_pos[0]: return True
    
    return False

def process_data_for_MTB(data):
    '''
    MTB를 위한 데이터 전처리 함수
    
    filter sentence 함수의 filtering 조건 또는
    entity 쌍에 대한 문장의 수가 2개보다 적은 경우(positive sentence pair를 만들 수가 없으므로) filtering
    
    Args:
        data: Original data for pre-training, key로 relation을 가지는 dictionary 형태
        
            data 예시:
                {
                    'P1': [
                        {
                            'token': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }
    Returns:
        반환하는 값은 없음
        json 형태의 file을 생성
            - list_data(mtbdata): 문장들이 담긴 list
            - entpair2scope: python dictionary 형태. 'head_id#tail_id'가 key로 존재하고
                value는 범위 인덱스가 나타나 있음. 같은 scope에 있는 모든 문장은 같은 entity pair로 이루어져 있음.
            - entpair2negpair: python dictionary 형태. 'head_id#tail_id'가 key로 존재하고
                value는 마찬가지로 'head_id#tail_id' 형태이지만 head 또는 tail 둘 중 하나만 다른 형태
                
            각 file 예시:
                - list_data(mtbdata):
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                - entpair2scope:
                    {
                        'Q1234#Q2356': [0, 233],
                        'Q135656#Q10': [233, 1000],
                        ...
                    }
                - entpair2negpair:
                    {
                        'Q1234#Q2356': ['Q1234#Q3560', 'Q923#Q2356', 'Q1234#Q100'],
                        'Q135656#Q10': ['Q135656#Q9', 'Q135656#Q10010', 'Q2666#Q10'],
                        ...
                    }
    Raises:
        data format이 위와 같은 형태가 아니면 key not found error를 발생시킴
    '''
    # 같은 entity pair를 가지는 문장의 최대 수.
    # 많은 문장 수를 가지는 popular entity pairs에 대해 bias를 갖게 되는 것을 막기 위해 설정.
    max_num = 8
    
    # 원래 데이터의 format을 바꿨음. ent_data는 python dictionary 형태로 key가 'head_id#tail_id' 형태이고,
    # value가 같은 entity pair를 가지는 문장들이다.
    ent_data = defaultdict(list)
    for key in data.keys():
        for sentence in data[key]:
            if filter_sentence(sentence):
                continue
            head = sentence["h"]["id"]
            tail = sentence["t"]["id"]
            ent_data[head + "#" + tail].append(sentence)
            
    ll = 0
    list_data = []
    entpair2scope = {}
    for key in ent_data.keys():
        if len(ent_data[key]) < 2:
            continue
        list_data.extend(ent_data[key][0:max_num])   # max_num 개수 만큼만 같은 entity pair를 가지는 문장을 가져옴
        entpair2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
        
    # 'hard' negative sample 만들기
    # entpair2negpair는 dictionary형태. key는 'head_id#tail_id'.
    # value는 head나 tail 중 하나의 id만 다른 pair들.
    entpair2negpair = defaultdict(list)
    entpairs = list(entpair2scope.keys())
    
    entpairs.sort(key=lambda x: x.split("#")[0])

    for i in range(len(entpairs)):
        head = entpairs[i].split("#")[0]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[0] != head:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])
    
    entpairs.sort(key=lambda x: x.split("#")[1])
    
    for i in range(len(entpairs)):
        tail = entpairs[i].split("#")[1]
        for j in range(i+1, len(entpairs)):
            if entpairs[j].split("#")[1] != tail:
                break
            entpair2negpair[entpairs[i]].append(entpairs[j])
    
    if not os.path.exists("./pretrain/data/MTB"):
        os.mkdir("./pretrain/data/MTB")
    json.dump(entpair2negpair, open("./pretrain/data/MTB/entpair2negpair.json", "w"))
    json.dump(entpair2scope, open("./pretrain/data/MTB/entpair2scope.json", "w"))
    json.dump(list_data, open("./pretrain/data/MTB/mtbdata.json", "w"))

def process_data_for_CP(data):
    '''
    CP를 위한 데이터 전처리 함수.
    NA relation인 문장, mention이 같거나 overlap되는 문장(abnormal sentences),
    relation에 대한 문장의 수가 2보다 적은 문장(positive sentence pair 불가능하므로)을 filtering.
    
    Args:
        data: pre-training을 위한 원래 데이터, key로 relation을 가지는 dictionary 형태
            data 예시:
                {
                    'P1': [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ],
                    ...
                }
    Returns:
        반환 값은 존재하지 않음.
        json format의 file을 2개 생성
            - list_data(cpdata): 문장을 담은 list
            - rel2scope: python dictionary 형태. relation이 key로 존재하고
                value는 범위 인덱스가 나타나 있음. 같은 scope에 있는 모든 문장은 같은 relation으로 이루어져 있음.
            
            각 file 예시:
                - list_data(cpdata):
                    [
                        {
                            'tokens': ['Microsoft', 'was', 'founded', 'by', 'Bill', 'Gates', '.']
                            'h': {'pos':[[0]], 'name': 'Microsoft', 'id': Q123456},
                            't': {'pos':[[4,5]], 'name': 'Bill Gates', 'id': Q2333},
                            'r': 'P1'
                        },
                        ...
                    ]
                - rel2scope:
                    {
                        'P10': [0, 233],
                        'P1212': [233, 1000],
                        ....
                    }
                    
    Raises:
        data의 형태가 맞지 않은 경우 key not found error 발생시킴
    '''
    washed_data = {}
    for key in data.keys():
        if key == "P0":
            continue
        rel_sentence_list = []
        for sen in data[key]:
            if filter_sentence(sen):
                continue
            rel_sentence_list.append(sen)
        if len(rel_sentence_list) < 2:
            continue
        washed_data[key] = rel_sentence_list
        
    ll = 0
    rel2scope = {}
    list_data = []
    for key in washed_data.keys():
        list_data.extend(washed_data[key])
        rel2scope[key] = [ll, len(list_data)]
        ll = len(list_data)
        
    if not os.path.exists("./pretrain/data/CP"):
        os.mkdir("./pretrain/data/CP")
    json.dump(rel2scope, open("./pretrain/data/CP/rel2scope.json", "w"))
    json.dump(list_data, open("./pretrain/data/CP/cpdata.json", "w"))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="MTB", help="{MTB, CP}")
    args = parser.parse_args()
    set_seed(42)

    data = json.load(open("../data/exclude_fewrel_distant.json"))
    if args.dataset == "CP":
        process_data_for_CP(data)
    elif args.dataset == "MTB":
        process_data_for_MTB(data)
        