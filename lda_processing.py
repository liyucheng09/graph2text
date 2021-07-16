#encoding: utf-8

import gensim
import json
from datasets import load_dataset
import sys
import math
import os
import pickle

num_worker=10

ds_cache_dir, worker_id, output_dir, lda_path, = sys.argv[1:]

ds = load_dataset('xsum', cache_dir=ds_cache_dir)
print(f'Loaded ds at {ds_cache_dir}')

ds_size = sum([len(v) for k,v in ds.items()])
each_party_size = math.ceil(ds_size/num_worker)

start_point = each_party_size * worker_id
end_point = start_point + each_party_size

lda = gensim.models.ldamulticore.LdaMulticore.load(lda_path, mmap='r')

count=0
for k,v in ds.items():
    for doc in v:
        count+=1
        if count<start_point:
            continue
        if count>end_point:
            print(f'== End at {count}. ')
            exit()
        doc_text = doc['document']
        doc_id = doc['id']
        bow = lda.id2word.doc2bow(doc_text.split())
        topics = lda.get_document_topics(bow, per_word_topics=1e-8, minimum_probability=1e-8, minimum_phi_value=1e-8)

        output_path = os.path.join(output_dir, k, doc_id + '.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(topics, f)
        print(f'Saved to {output_path}')
