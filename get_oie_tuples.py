from datasets import load_dataset
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import pickle
predictor = Predictor.from_path("models/openie-model.2020.03.26.tar.gz")


ds=load_dataset('xsum')
processed=[]
for i in ds['validation']:
    document = i['document']
    summary = i['summary']
    doc_verbs = predictor.predict(sentence=document)['verbs']
    summary_verbs = predictor.predict(sentence=summary)['verbs']
    i['doc_verbs'] = doc_verbs
    i['summary_verbs'] = summary_verbs
    processed.append(i)

with open('OIEed_val_xsum.pkl', 'wb') as f:
    pickle.dump(processed, f)

print('Saved to pickle.')