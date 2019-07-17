#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:28:30 2019

@author: vaibhav
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

import pandas as pd
import numpy as np

from sklearn.tree.export import export_text

from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/Users/vaibhav/MiscProjects/test_data.csv')

df2 = df[['hobby','age','edu','mars','class']]

df2['class'] = df2['class'] - 1

enc = OneHotEncoder(handle_unknown='ignore')

enc = enc.fit(df2[['hobby','age','edu','mars']])

x = enc.transform(df2[['hobby','age','edu','mars']])

clf = DecisionTreeClassifier(random_state=0, max_depth=6)
clf = clf.fit(x, df2['class'].values)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto=dict()

    global k
    k = 0
    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s= "{} <= {}".format(name, threshold, node)
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+'|' +s

            recurse(tree_.children_left[node], depth + 1, node)
            s="{} > {}".format(name, threshold)
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+'|' +s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k=k+1
            print(k,')',pathto[parent], tree_.value[node])
    recurse(0, 1, 0)
    
tree_to_code(clf, list(enc.get_feature_names()))

r = export_text(clf, feature_names=list(enc.get_feature_names()))

print(r)


rule = 'x2_4 <= 0.5|x3_4 <= 0.5|x1_4 <= 0.5|x3_2 <= 0.5|x1_1 <= 0.5|x2_2 <= 0.5'

rule_parts = rule.split('|')

rule_new = ''

for rp in rule_parts:
    if '<= 0.5' in rp:
        rp1 = rp.split(' <= ')[0].split('_')
        rule_new = rule_new + rp1[0] + ' != ' + rp1[1] + '|'
    elif '> 0.5' in rp:
        rp1 = rp.split(' > ')[0].split('_')
        rule_new = rule_new + rp1[0] + ' != ' + rp1[1] + '|'
        










