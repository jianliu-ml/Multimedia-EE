import os
from tkinter import X
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import pickle

import config
from dataset_arg import Dataset
from model_arg import BertNER
from conlleval import evaluate_conll_file
from utils import save_model, load_model

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

idx2tag = config.idx2tag

def load_dataset(filename):
    data = pickle.load(open(filename, 'rb'))
    return data

def load_conll_data(fpath):
    results = []

    words, tags_g, tags_p = list(), list(), list()
    for line in open(fpath, 'r'):
        line = line.strip()
        if len(line) == 0:
            if words:
                results.append([words, tags_g, tags_p])
                words, tags_g, tags_p = list(), list(), list()
        else:
            w = line.split(' ')[0]
            t1 = line.split(' ')[1]
            t2 = line.split(' ')[2]
            words.append(w)
            tags_g.append(t1)
            tags_p.append(t2)
    
    if words:
        results.append([words, tags_g, tags_p])
    return results


if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 20
    n_epochs = 20

    train, dev, test, test_data = load_dataset('data/data_arg.pk')
    train = train + dev + test
    print(train[0])

    sequence_len = 200

    train_dataset = Dataset(batch_size, sequence_len, train)
    test_dataset = Dataset(batch_size, sequence_len, test_data)   # <---- dev data

    model = BertNER(config.bert_dir, len(config.idx2tag_role))
    model.to(device)

    num_warmup_steps = 0
    num_training_steps = n_epochs * (len(train) / batch_size)
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

    # load_model(model, 'models/model_9')
    for epo in range(n_epochs):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, bert_mask, data_span, data_y = batch
            loss = model(data_x, bert_mask, data_span, data_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        save_model(model, 'models/model_%d' % (epo))


        model.eval()

        p_count, g_count, correct_count = 0, 0, 0

        trigger_result = load_conll_data('predict_ref.conll')

        for x, trigger_res in zip(test_data, trigger_result):
            d_x, span, _, ev, entity = x
            g = list()
            for e in ev:
                for a in e['arguments']:
                    g.append((e['event_type'].split(':')[1], span[e['trigger']['start']][0], span[a['start']][0], a['role']))
            
            

            # print(entity)
            _, _, predicted = trigger_res
            my_span = list()
            ev_types = list()
            for idx, pred in enumerate(predicted):
                if pred != 'O':
                    for e in entity:
                        # print(span[idx][0], pred, span[e['start']][0])
                        my_span.append((span[idx][0], span[e['start']][0]))
                        ev_types.append(pred[2:])
            xxx = list()
            if my_span:
                f = torch.LongTensor
                d_x = f(d_x).unsqueeze(0).to('cuda')
                ms = torch.ones(d_x.size()).to('cuda')
                sp = [my_span]
                
                xxx = model.predict(d_x, ms, sp)

            p = list()
            for ms, et, x in zip(my_span, ev_types, xxx):
                if config.idx2tag_role[x] != 'O':
                    l = (et, ms[0], ms[1], config.idx2tag_role[x])
                    p.append(l)
            # print('G', g)
            # print('P', p)

            g_count += len(g)
            p_count += len(p)
            correct_count += len(set(g).intersection(set(p)))
            # print(my_span)
        
        p = correct_count / p_count
        r = correct_count / g_count
        f1 = 2 * p * r / (p + r)
        print(p, r, f1)


        # evaluate
        # model.eval()
        # all_words = []
        # all_labels = []
        # all_predicts = []
        # for batch in test_dataset.get_tqdm(device, False):
        #     data_x, bert_mask, data_span, sequence_mask, data_y, words, labels = batch
        #     predicts = model.predict(data_x, bert_mask, data_span, sequence_mask)
        #     all_predicts.extend(predicts)
        #     all_words.extend(words)
        #     all_labels.extend(labels)


        # def _transfer(l, p, idx2tag):

        #     if l == 'O':
        #         l = 'O'
        #     else:
        #         l = 'B-' + l.replace('-', '_')
        #         # l = 'B-' + config.subtype_to_type[l]
            
        #     p = idx2tag[p]

        #     if p == 'O':
        #         p = 'O'
        #     else:
        #         p = 'B-' + p.replace('-', '_')
        #         # p = 'B-' + config.subtype_to_type[p]

        #     return l, p
        
        
        # filename = 'predict.conll'
        # fileout = open(filename, 'w')
        # for words, labels, predicts in zip(all_words, all_labels, all_predicts):
        #     for w, l, p in zip(words, labels, predicts):
        #         l, p = _transfer(l, p, idx2tag)
        #         print(w, l, p, file=fileout)
        #     print(file=fileout)
        # fileout.close()

        # with open(filename) as fout:
        #     evaluate_conll_file(fout) 