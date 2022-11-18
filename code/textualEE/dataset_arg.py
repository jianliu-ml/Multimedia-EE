import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        sub_word_lens = [len(tup[0]) for tup in batch]
        max_sub_word_len = self.seq_len

        data_x, data_span, data_y = list(), list(), list()

        for data in batch:
            data_x.append(data[0])

            xxx, yyy = list(), list()

            for i in range(len(data[1])):
                if data[1][i][0] < max_sub_word_len and data[1][i][1] < max_sub_word_len:
                    xxx.append(data[1][i])
                    yyy.append(data[2][i])
            if not xxx:
                xxx.append((0, 0))
                yyy.append(0)
            data_span.append(xxx)
            data_y.append(yyy)

        f = torch.LongTensor

        data_x = list(map(lambda x: pad_sequence_to_length(x, max_sub_word_len), data_x))
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        return [f(data_x).to(device),  
                bert_mask.to(device),
                data_span,
                data_y
                ]


if __name__ == "__main__":
    from preprocess import read_ace, build_bert_examples
    
    result = read_ace('data/train.json', True)
    examples = build_bert_examples(result)
    
    train_dataset = Dataset(5, 200, examples)

    for batch in train_dataset.reader('cpu', False):
        data_x, bert_mask, data_span, sequence_mask, data_y, words, labels = batch
        print(data_x[0])
        print(bert_mask[0])
        print(data_y[0])
        break