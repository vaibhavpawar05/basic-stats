#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:00:32 2019

@author: vaibhav
"""

# classes 2, categories - 3 to 20
# sample random conuts for classes*categories cells
# find best binary split - gini, crossentropy
# brute force vs the ordering trick
# compare

import numpy as np
import itertools as it
import math

def nCr(n,r):
    f = math.factorial
    return f(n) // (f(r)*f(n-r))

# find best subset for minimizing CE

def find_bs_bruteforce(ncat, counts, counts_cat, n, n0):

    subsets = []
    scores = []
                
    for i in range(1, (ncat // 2) + 1):
        if ncat % 2 == 0 and i == (ncat // 2):
            max_comb = nCr(ncat, i) // 2
            j = 0
            for c in it.combinations(range(ncat), i):
                # calc cross entropy
                left_idx = list(c)
                left_n = np.sum(counts_cat[left_idx])
                left_n0 = np.sum(counts[0, left_idx])
                left_p0 = left_n0/left_n
                left_ce = -1 * ((left_p0 * np.log(left_p0) + 
                                (1 - left_p0) * np.log(1 - left_p0)))
                
                right_n = n - left_n
                right_n0 = n0 - left_n0
                right_p0 = right_n0/right_n
                right_ce = -1 * ((right_p0 * np.log(right_p0) + 
                                (1 - right_p0) * np.log(1 - right_p0)))
                
                ce = (left_n / n) * left_ce + (right_n / n) * right_ce
                     
                subsets.append(left_idx)
                scores.append(ce)
                
                j += 1
                if j >= max_comb:
                    break
        else:
            for c in it.combinations(range(ncat), i):
                #print(c)
                left_idx = list(c)
                left_n = np.sum(counts_cat[left_idx])
                left_n0 = np.sum(counts[0, left_idx])
                left_p0 = left_n0/left_n
                left_ce = -1 * ((left_p0 * np.log(left_p0) + 
                                (1 - left_p0) * np.log(1 - left_p0)))
                
                right_n = n - left_n
                right_n0 = n0 - left_n0
                right_p0 = right_n0/right_n
                right_ce = -1 * ((right_p0 * np.log(right_p0) + 
                                (1 - right_p0) * np.log(1 - right_p0)))
                
                ce = (left_n / n) * left_ce + (right_n / n) * right_ce
                     
                subsets.append(left_idx)
                scores.append(ce)
                
    min_ce = np.min(scores)
    min_idx = [i for i, v in enumerate(scores) if v == min_ce]
    
    best_subsets = [subsets[i] for i in min_idx]
    
    return min_ce, best_subsets



def find_bs_heuristic(ncat, counts, counts_cat, n, n0, p0_cat):
    
    # now find best split using ordering by p0

    sort_idx = np.argsort(-p0_cat)
    
    subsets = []
    scores = []
    
    for i in range(1, ncat):
        left_idx = list(sort_idx[0:i])
        left_idx.sort()
        left_n = np.sum(counts_cat[left_idx])
        left_n0 = np.sum(counts[0, left_idx])
        left_p0 = left_n0/left_n
        left_ce = -1 * ((left_p0 * np.log(left_p0) + 
                        (1 - left_p0) * np.log(1 - left_p0)))
        
        right_n = n - left_n
        right_n0 = n0 - left_n0
        right_p0 = right_n0/right_n
        right_ce = -1 * ((right_p0 * np.log(right_p0) + 
                        (1 - right_p0) * np.log(1 - right_p0)))
        
        ce = (left_n / n) * left_ce + (right_n / n) * right_ce
             
        subsets.append(left_idx)
        scores.append(ce)
                
    min_ce = np.min(scores)
    min_idx = [i for i, v in enumerate(scores) if v == min_ce]
    
    best_subsets = [subsets[i] for i in min_idx]
    
    return min_ce, best_subsets

nclass = 2
ntests = 10
maxcat = 20
maxcount = 1000

for i in range(0, ntests):
    
    ncat = np.random.randint(low=3, high=maxcat)

    # randomly sample integers for each cell - 1 to 1000
    # counts = np.zeros((nclass, ncat), dtype='int')
    counts = np.random.randint(low=0, high=maxcount, size=(nclass, ncat), dtype='int')

    # ensure that every category has at least 1 record in total
    # category wise counts
    counts_cat = np.sum(counts, axis=0)

    for i in range(0, ncat):
        if counts_cat[i] == 0:
            counts[1, i] = 1

    counts_cat = np.sum(counts, axis=0)
    n = np.sum(counts_cat)
            
    # class wise conuts
    counts_class = np.sum(counts, axis=1)
    n0 = counts_class[0]
    
    # p0
    p0 = n0/n
    p0_cat = np.divide(counts[0, :], counts_cat)

    # CE before split
    ce = -1 * ((p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
    
    # find best subset for minimizing CE
    
    min_ce1, best_subsets1 = find_bs_bruteforce(ncat, counts, 
                                                counts_cat, n, n0)
    
    # now find best split using ordering by p0
    
    min_ce2, best_subsets2 = find_bs_heuristic(ncat, counts, counts_cat, 
                                               n, n0, p0_cat)
    
    #print(ncat, '|', min_ce1, '-', best_subsets1, '|', 
    #      min_ce2, '-', best_subsets2)
    
    print('bf |', min_ce1, '-', best_subsets1)
    print('hr |', min_ce2, '-', best_subsets2)
    print()
