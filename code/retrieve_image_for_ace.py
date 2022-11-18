import json
from gensim import corpora
from gensim.summarization import bm25


voa_data = json.loads(open('data/voa_img_dataset.json').read())
docs = list()
for k in voa_data:
    a_doc = list()
    for l in voa_data[k]:
        a_doc += voa_data[k][l]['cap'].split()
    a_doc.append(k)
    docs.append(' '.join(a_doc))



texts = [doc.split() for doc in docs] # you can do preprocessing as removing stopwords
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
bm25_obj = bm25.BM25(corpus)

s_set = set()

filelines = open('data/ace/train.json').readlines() + open('data/ace/dev.json').readlines() + open('data/ace/test.json').readlines()
    
for l in filelines:
    zz = json.loads(l)
    sentences = zz['sentences']
    flat_list = [item for sublist in sentences for item in sublist]

    query_doc = dictionary.doc2bow(flat_list)
    scores = bm25_obj.get_scores(query_doc)
    best_docs = sorted(range(len(scores)), key=lambda i: scores[i])[-3:]
    best_docs = best_docs[::-1]

    ll = [docs[x].split()[-1] for x in best_docs]
    print(zz['doc_key'], ' '.join(ll))

    for z in ll:
        s_set.add(z)

for k in voa_data.copy():
    if k not in s_set:
        del voa_data[k]

with open('voa_filter.json', 'w') as outfile:
    json.dump(voa_data, outfile)
