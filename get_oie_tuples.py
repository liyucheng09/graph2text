from datasets import load_dataset
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pickle
import sys
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    num_gpu = 0
else:
    num_gpu = -1

model_path, output_file, = sys.argv[1:]
print(f'Model path: {model_path}')

predictor = Predictor.from_path(model_path, cuda_device = num_gpu)
print('Loaded predictor!')
ds=load_dataset('xsum')
processed=[]
try:
    for i in tqdm(ds['validation']):
        document = i['document']
        summary = i['summary']
        doc_verbs = predictor.predict(sentence=document)['verbs']
        summary_verbs = predictor.predict(sentence=summary)['verbs']
        i['doc_verbs'] = doc_verbs
        i['summary_verbs'] = summary_verbs
        processed.append(i)
finally:
    with open(output_file, 'wb') as f:
        pickle.dump(processed, f)

    print(f'Saved to {output_file}.')