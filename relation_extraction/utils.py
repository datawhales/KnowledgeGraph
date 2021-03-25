import os
import re
import pdb
import ast
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from transformers import BertTokenizer
from collections import Counter, defaultdict

class EntityMarker:
    """raw text를 BERT-input ids로 바꾸고 entity position을 찾는 클래스.
    
    Attributes:
        tokenizer: Bert-base tokenizer
        h_pattern: 정규표현식 패턴 -- * h * 이용. head entity mention을 대체하는데 이용.
        t_pattern: 정규표현식 패턴 -- ^ t ^ 이용. tail entity mention을 대체하는데 이용.
        err: 정상적으로 head/tail entity를 찾을 수 없는 문장의 개수를 기록
        args: command line으로부터의 args    
    """
    def __init__(self, args=None):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args
        
    def tokenize(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False):
        """C+M, C+T, OnlyC setting의 tokenize 함수.
        
        raw text를 BERT-input ids로 바꾸고, entity-marker를 이용하여 entity 위치를 표시한 뒤,
        random하게 entity mention을 [BLANK] symbol로 바꿔준다.
        Entity mention은 entity type이 될 수도 있다.
        반환하는 값은 BERT에 들어갈 input-ids, entity position.
        
        Args:
            raw_text: tokens가 담긴 리스트.
            h_pos_li: head entity position을 담은 리스트. ex) head entity mention이 raw_text[2:6]이면 h_pos_li = [2, 6]
            t_pos_li: tail entity position을 담은 리스트.
            h_type: head entity type. C+T 세팅 시 이용.
            t_type: tail entity type. C+T 세팅 시 이용.
            h_blank: head entity mention을 [BLANK]로 바꿀지 말지 여부.
            t_blank: tail entity mention을 [BLANK]로 바꿀지 말지 여부.
        
        Returns:
            tokenized_input: BERT에 바로 들어갈 수 있는 input-ids 형태.
            h_pos: head entity marker start position
            t_pos: tail entity marker start position
        
        예시:
            raw_text: ["Bill", "Gates", "founded", "Microsoft", "."]
            h_pos_li: [0, 2]
            t_pos_li: [3, 4]
            h_type: None
            t_type: None
            h_blank: True
            t_blank: False
            
            1. entity mention을 special pattern으로 대체해준다:
            "* h * founded ^ t ^ ."
            
            2. pattern을 대체해준다:
            "[CLS] [unused0] [unused4] [unused1] founded [unused2] microsoft [unused3] . [SEP]"
            
            3. input id로 변환 및 entity marker start position 찾는다:
            [101, 1, 5, 2, 2631, 3, 7513, 4, 1012, 102]
            h_pos: 1, t_pos: 5
        """
        tokens = []
        h_mention = []
        t_mention = []
        
        for i, token in enumerate(raw_text):
            token = token.lower()
            if i >= h_pos_li[0] and i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_li[0] and i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)
        
        # tokenize
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_head = self.tokenizer.tokenize(h_mention)
        tokenized_tail = self.tokenizer.tokenize(t_mention)
        
        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)
        
        # head entity type과 tail entity type이 None이 아니라면,
        # C+T setting을 이용 -> entity mention을 entity type으로 대체
        if h_type != None and t_type != None:
            p_head = h_type
            p_tail = t_type
            
        # h_blank와 t_blank가 각각 True이면 entity mention을 blank로 대체
        if h_blank:
            p_text = self.h_pattern.sub("[unused0] [unused4] [unused1]", p_text)
        else:
            p_text = self.h_pattern.sub("[unused0] " + p_head + " [unused1]", p_text)
        if t_blank:
            p_text = self.t_pattern.sub("[unused2] [unused5] [unused3]", p_text)
        else:
            p_text = self.t_pattern.sub("[unused2] " + p_tail + " [unused3]", p_text)
            
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        
        # 만약 h_pos_li와 t_pos_li에서 overlap이 발생하면, head entity와 tail entity를 제대로 찾을 수 없음
        try:
            h_pos = f_text.index("[unused0]")
        except:
            self.err += 1
            h_pos = 0
        try:
            t_pos = f_text.index("[unused2]")
        except:
            self.err += 1
            t_pos = 0
            
        tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        
        return tokenized_input, h_pos, t_pos
    
    def tokenize_OMOT(self, tokenized_head, tokenized_tail, h_first):
        '''OnlyM, OnlyT setting의 tokenize 함수.
        
        head entity와 tail entity를 id로 바꿔준다.
        
        Args:
            tokenized_head: Head entity mention 또는 type을 리스트 형태로 담고 있음. BertTokenizer로 tokenized.
            tokenized_tail: Tail entity mention 또는 type을 리스트 형태로 담고 있음. BertTOkenizer로 tokenized.
            h_first: head entity가 첫 번째 entity인지의 여부
            
        Returns:
            tokenized_input: BERT에 바로 들어갈 수 있는 input-ids 형태.
            h_pos: head entity marker start position
            t_pos: tail entity marker start position
        '''
        
        tokens = ["[CLS]",]
        
        if h_first:
            h_pos = 1
            tokens += ["[unused0]",] + tokenized_head + ["[unused1]",]
            t_pos = len(tokens)
            tokens += ["[unused2]",] + tokenized_tail + ["[unused3]",]
        else:
            t_pos = 1
            tokens += ["[unused2]",] + tokenized_tail + ["[unused3]",]
            h_pos = len(tokens)
            tokens += ["[unused0]",] + tokenized_head + ["[unused1]",]
            
        tokens.append("[SEP]")
        
        tokenized_input = tokenizer.convert_tokens_to_ids(tokens)
            
        return tokenized_input, h_pos, t_pos