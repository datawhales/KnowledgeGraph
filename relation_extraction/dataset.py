import json
import random
import os
import sys
sys.path.append("..")
import pdb
import re
import math
import torch
import numpy as np
from collections import Counter
sys.path.append('../../')
from utils import EntityMarker

class CPDataset(torch.utils.data.Dataset):
    """CP 학습을 위한 데이터셋 클래스.
    
    """
    def __init__(self, path, args):
        """tokenized sentence 초기화, CP를 위한 positive pair 생성
        
        Args:
            path: dataset 경로
            args: command line args
        
        Returns:
            반환하는 값은 존재하지 않음
        
        Raises:
            경로에 있는 dataset이 prepare_data.py에 나타난 형태가 아니면,
                - 'key not found'
                - 'integer can't be indexed'
                와 같은 에러 발생
        """
        self.path = path
        self.args = args
        data = json.load(open(os.path.join(path, "cpdata.json")))
        rel2scope = json.load(open(os.path.join(path, "rel2scope.json")))
        entityMarker = EntityMarker()
        
        self.tokens = np.zeros((len(data), args.max_length), dtype=int)
        self.mask = np.zeros((len(data), args.max_length), dtype=int)
        self.label = np.zeros((len(data)), dtype=int)
        self.h_pos = np.zeros((len(data)), dtype=int)
        self.t_pos = np.zeros((len(data)), dtype=int)
        
        # distant supervised label
        # label이 같은 문장은 positive pair, 그렇지 않으면 negative pair
        for i, rel in enumerate(rel2scope.keys()):
            scope = rel2scope[rel]
            for j in range(scope[0], scope[1]):
                self.label[j] = i
                
        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]     # [10,11,12] 형태
            t_p = sentence["t"]["pos"][0]     # [18, 19] 형태
            ids, ph, pt = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1], None, None, h_flag, t_flag)
            length = min(len(ids), args.max_length)
            self.tokens[i][:length] = ids[:length]
            self.mask[i][:length] = 1
            self.h_pos[i] = min(args.max_length - 1, ph)
            self.t_pos[i] = min(args.max_length - 1, pt)
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)
        
        # sample positive pair dynamically
        self.__sample__()
    
    def __pos_pair__(self, scope):
        """positive pair 생성
        
        Args:
            scope: label이 같은 문장의 인덱스 범위
                example: [0, 12]
        
        Returns:
            all_pos_pair: 모든 positive pairs를 반환.
            
            ********
            같은 범위 안에 존재하는 문장 쌍은 모두 positive pair이므로,
            N = scope[1] - scope[0]이라 할 때 (N-1)N/2개의 쌍이 존재.
            개수가 N^2에 비례하므로 데이터 사이의 불균형이 발생.
            이를 해결하기 위해 N에 비례하도록 positive pair를 sampling한다.
            epoch이 달라질 때마다 sentence pair를 다시 sampling, i.e. dynamic sampling.
        """
        pos_scope = list(range(scope[0], scope[1]))
        
        # shuffle
        random.shuffle(pos_scope)
        all_pos_pair = []
        bag = []
        for i, index in enumerate(pos_scope):
            bag.append(index)
            if (i+1) % 2 == 0:
                all_pos_pair.append(bag)
                bag = []
        return all_pos_pair
        
    def __sample__(self):
        """Samples positive pairs.
        
        Sampling 후에, 'self.pos_pair'는 all pairs sampled.
        'self.pos_pair' example:
            [
                [0, 2],
                [1, 6],
                [12, 25],
                ...
            ]
        
        """
        rel2scope = json.load(open(os.path.join(self.path, "rel2scope.json")))
        self.pos_pair = []
        for rel in rel2scope.keys():
            scope = rel2scope[rel]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)
        
        print("Positive pair's number is %d" % len(self.pos_pair))
        
    def __len__(self):
        """Number of instances in an epoch.        
        """
        return len(self.pos_pair)
        
    def __getitem__(self, index):
        """Get training instance.
        
        Args:
            index: Instance index.
            
        Returns:
            input: Tokenized word id
            mask: Attention mask for bert. 0 means masking, 1 means not masking
            label: label for sentence
            h_pos: head entity marker 위치
            t_pos: tail entity marker 위치
        """
        bag = self.pos_pair[index]
        input = np.zeros(self.args.max_length * 2, dtype=int)
        mask = np.zeros(self.args.max_length * 2, dtype=int)
        label = np.zeros(2, dtype=int)
        h_pos = np.zeros(2, dtype=int)
        t_pos = np.zeros(2, dtype=int)
        
        for i, ind in enumerate(bag):
            input[i * self.args.max_length:(i+1) * self.args.max_length] = self.tokens[ind]
            mask[i * self.args.max_length:(i+1) * self.args.max_length] = self.mask[ind]
            label[i] = self.label[ind]
            h_pos[i] = self.h_pos[ind]
            t_pos[i] = self.t_pos[ind]
        
        return input, mask, label, h_pos, t_pos
        
class MTBDataset(torch.utils.data.Dataset):
    """MTB 학습을 위한 데이터셋 클래스.
    """
    def __init__(self, path, args):
        """Tokenized 문장 초기화 및 MTB를 위한 positive pair 생성.
        
        Args:
            path: dataset 경로
            args: command line args
        
        Returns:
            반환하는 값은 존재하지 않음
            
        Raises:
            경로에 있는 dataset이 prepare_data.py에 나타난 형태가 아니면,
                - 'key not found'
                - 'integer can't be indexed'
                와 같은 에러 발생
        """
        self.path = path
        self.args = args
        data = json.load(open(os.path.join(path, "mtbdata.json")))
        entityMarker = EntityMarker()
        
        # important configures
        tot_sentence = len(data)
        
        
        # token들을 id로 바꾸고 몇 개의 entity를 random하게 blank로 바꿔준다.
        self.tokens = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.mask = np.zeros((tot_sentence, args.max_length), dtype=int)
        self.h_pos = np.zeros(tot_sentence, dtype=int)
        self.t_pos = np.zeros(tot_sentence, dtype=int)
        
        for i, sentence in enumerate(data):
            h_flag = random.random() > args.alpha
            t_flag = random.random() > args.alpha
            h_p = sentence["h"]["pos"][0]
            t_p = sentence["t"]["pos"][0]
            ids, ph, pt = entityMarker.tokenize(sentence["tokens"], [h_p[0], h_p[-1]+1], [t_p[0], t_p[-1]+1],
                                                None, None, h_flag, t_flag)
            length = min(len(ids), args.max_length)
            self.tokens[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(args.max_length - 1, ph)
            self.t_pos[i] = min(args.max_length - 1, pt)
            
        print("The number of sentence in which tokenizer can't find head/tail entity is %d" % entityMarker.err)

        entpair2scope = json.load(open(os.path.join(path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(path, "entpair2negpair.json")))
        self.pos_pair = []
        
        for key in entpair2scope.keys():
            scope = entpair2scope[key]
            pos_pair = self.__pos_pair__(scope)
            self.pos_pair.extend(pos_pair)
        print("Positive pairs' number is %d" % len(self.pos_pair))
        
        # sample negative pairs dynamically
        self.__sample__()
        
    def __sample__(self):
        """negative pairs를 sampling하는 함수.
        
        entpair2negpair는 dictionary 형태로 key가 head_id#tail_id 형태이고,
        value는 head나 entity 둘 중 하나만 다른 형태
        
        *********
        negative pair의 수가 positive pair의 수와 같은 만큼 sampling 수행
        """
        entpair2scope = json.load(open(os.path.join(path, "entpair2scope.json")))
        entpair2negpair = json.load(open(os.path.join(path, "entpair2negpair.json")))
        neg_pair = []
        
        # get all negative pairs
        for key in entpair2negpair.keys():
            my_scope = entpair2scope[key]
            entpairs = entpair2negpair[key]
            if len(entpairs) == 0:
                continue
            for entpair in entpairs:
                neg_scope = entpair2scope[entpair]
                neg_pair.extend(self.__neg_pair__(my_scope, neg_scope))
        print("(MTB)Negative pairs number is %d" % len(neg_pair))
        
        # positive pair와 같은 수만큼 negative pair sampling
        random.shuffle(neg_pair)
        self.neg_pair = neg_pair[0:len(self.pos_pair)]
        del neg_pair   # save the memory
          
    def __pos_pair__(self, scope):
        """하나의 scope에 대해 positive pair를 생성하는 함수.
        
        Args:
            scope: 같은 entity pair를 가지는 문장의 scope
        
        Returns:
            pos_pair: scope 안에 있는 모든 positive pair를 반환.
        """
        ent_scope = list(range(scope[0], scope[1]))
        pos_pair = []
        
        for i in range(len(ent_scope)):
            for j in range(i+1, len(ent_scope)):
                pos_pair.append([ent_scope[i], ent_scope[j]])
        return pos_pair   
    
    def __neg_pair__(self, my_scope, neg_scope):
        """다른 scope에 있는 negative pair를 생성하는 함수.
        
        Args:
            my_scope: negative pair에 대해 기준이 되는 문장이 담긴 scope
            neg_scope: negative pair들이 모두 담긴 scope
        
        Returns:
            neg_pair: 모든 negative pairs의 scope를 반환.
        """
        my_scope = list(range(my_scope[0], my_scope[1]))
        neg_scope = list(range(neg_scope[0], neg_scope[1]))
        neg_pair = []
        for i in my_scope:
            for j in neg_scope:
                neg_pair.append([i, j])
        return neg_pair    
    
    def __len__(self):
        """Number of instances in an epoch.        
        """
        return len(self.pos_pair)
    
    def __getitem__(self, index):
        """Gets training instance.
        
        index가 홀수이면, negative instance를 반환하고 짝수이면 positive instance를 반환.
        batch에서는 positive pairs의 수와 negative pairs의 수가 같아짐.
        
        Args:
            index: Data index
            
        Returns:
            {l,h}_input: Tokenized word id.
            {l,h}_mask: Attention mask for bert.
            {l,h}_ph: head entity 위치
            {l,h}_pt: tail entity 위치
            label: positive 또는 negative
            
        """
        if index % 2 == 0:
            l_ind = self.pos_pair[index][0]
            r_ind = self.pos_pair[index][1]
            label = 1
        else:
            l_ind = self.neg_pair[index][0]
            r_ind = self.neg_pair[index][1]
            label = 0
        
        l_input = self.tokens[l_ind]
        l_mask = self.mask[l_ind]
        l_ph = self.h_pos[l_ind]
        l_pt = self.t_pos[l_ind]
        r_input = self.tokens[r_ind]
        r_mask = self.mask[r_ind]
        r_ph = self.h_pos[r_ind]
        r_pt = self.t_pos[r_ind]
        
        return l_input, l_mask, l_ph, l_pt, r_input, r_mask, r_ph, r_pt, label