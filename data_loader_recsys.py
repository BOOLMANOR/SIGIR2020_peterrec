import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in positive_examples])
        #如果是文档的话每个文档长度可能是不一样的，我们以每个最长的词数的段落生成多个一维向量
        #max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        #这里给每个数据打表相当于，13373出现在了第一个位置，那么在后面再出现13373，直接给他赋值成1
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        #变形
        self.item_dict = vocab_processor.vocabulary_._mapping
        #这里是字典，可以输出所有一一对应关系


        # added to calculate word frequency
        # allitems_hassamewords=list()
        # for line in self.item:
        #     for ele in line:
        #         allitems_hassamewords.append(ele)
        #
        # counts = Counter(allitems_hassamewords)
        # most_com=counts.most_common(10)
        # print allitems_hassamewords




