# -*- coding: utf-8 -*-
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import pandas as pd
import gensim
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings(action='ignore')
data_raw=pd.read_csv('./input/raw_data.csv',sep='\t',header=None,names=['id','question1','question2','is_dup'])

def extract_questions():
    """
    Extract questions for making word2vec model.
    """

    for dataset in [data_raw]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['question1']:
                yield gensim.utils.simple_preprocess(row['question1'])
            if row['question2']:
                yield gensim.utils.simple_preprocess(row['question2'])


documents = list(extract_questions())
logging.info("Done reading data file")

model = gensim.models.Word2Vec(documents, size=300)
model.train(documents, total_examples=len(documents), epochs=10)
model.save("./input/Quora.w2v")




