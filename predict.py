# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import keras
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths

# Load training set
test_df = pd.read_csv(sys.argv[1],sep='\t',header=None,names=['id','question1','question2'])
#test_df = pd.read_csv('./input/test.csv',sep='\t',header=None,names=['id','question1','question2'])
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)
# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape
model = keras.models.load_model('./input/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()
prediction = model.predict([X_test['left'], X_test['right']])

with open(sys.argv[2],'w') as fin:
    for i,probs in enumerate(prediction):
        if probs[0]>0.5:
            fin.write(str(i)+'\t'+str(1)+'\n')
        else:
            fin.write(str(i)+'\t'+str(0)+'\n')
# with open('./input/output.txt','w') as fin:
#     for i,probs in enumerate(prediction):
#         if probs[0]>0.5:
#             fin.write(str(i)+'\t'+str(1)+'\n')
#         else:
#             fin.write(str(i)+'\t'+str(0)+'\n')
