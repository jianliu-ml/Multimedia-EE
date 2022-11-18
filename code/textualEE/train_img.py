import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import pickle
import config

from dataset_img import Dataset
from model_img import BertNER
from conlleval import evaluate_conll_file
from utils import save_model
print('Here')
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


idx2tag = config.idx2tag

def load_dataset(filename):
    data = pickle.load(open(filename, 'rb'))
    return data


if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 20
    n_epochs = 10

    train, dev, test, test_data = load_dataset('data/data.pk')
    
    print(train[0])
    print(train[1])
    print(train[2])
    print(train[3])

    sequence_len = 180

    train_dataset = Dataset(batch_size, sequence_len, train)
    test_dataset = Dataset(batch_size, sequence_len, test_data)   # <---- dev data

    model = BertNER(config.bert_dir, len(config.idx2tag))
    model.to(device)

    num_warmup_steps = 0
    num_training_steps = n_epochs * (len(train) / batch_size)
    warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = AdamW(parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

    for epo in range(n_epochs):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            data_x, bert_mask, data_span, sequence_mask, data_y, image, text, words, labels = batch
            loss = model(data_x, bert_mask, data_span, sequence_mask, data_y, image, text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        save_model(model, 'models/model_%d' % (epo))

        # evaluate
        model.eval()
        all_words = []
        all_labels = []
        all_predicts = []
        for batch in test_dataset.get_tqdm(device, False):
            data_x, bert_mask, data_span, sequence_mask, data_y, image, text, words, labels = batch
            predicts = model.predict(data_x, bert_mask, data_span, sequence_mask, image, text)
            all_predicts.extend(predicts)
            all_words.extend(words)
            all_labels.extend(labels)


        def _transfer(l, p, idx2tag):

            if l == 'O':
                l = 'O'
            else:
                l = 'B-' + l.replace('-', '_')
                # l = 'B-' + config.subtype_to_type[l]
            
            p = idx2tag[p]

            if p == 'O':
                p = 'O'
            else:
                p = 'B-' + p.replace('-', '_')
                # p = 'B-' + config.subtype_to_type[p]

            return l, p
        
        
        filename = 'predict.conll'
        fileout = open(filename, 'w')
        for words, labels, predicts in zip(all_words, all_labels, all_predicts):
            for w, l, p in zip(words, labels, predicts):
                l, p = _transfer(l, p, idx2tag)
                print(w, l, p, file=fileout)
            print(file=fileout)
        fileout.close()

        with open(filename) as fout:
            evaluate_conll_file(fout) 