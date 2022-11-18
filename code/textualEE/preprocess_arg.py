import json
import pickle
from re import L
from PIL import Image

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import os.path
import config

import glob
all_images = glob.glob("/home/jliu/research/5.MMEE/textualEE/ACE_Image/*")

tokenizer = config.tokenizer
tag2idx = config.tag2idx

file_set = set()
with open('flist') as filein:
    for line in filein:
        file_set.add(line.strip())


ace_image = {}
with open('ace_candidate_image.txt') as filein:
    for line in filein:
        field = line.strip().split()
        ace_image[field[0]] = field[1:]


def read_m2e2():
    file1 = '/home/jliu/research/5.MMEE/data/m2e2_annotations/m2e2_annotations/text_only_event.json'
    file2 = '/home/jliu/research/5.MMEE/data/m2e2_annotations/m2e2_annotations/text_multimedia_event.json'

    temp = []

    for f in [file1, file2]:
        xxx = json.loads(open(f).read())
        for x in xxx:
            sentence = x['words']
            event = x['golden-event-mentions']
            image = x['image']
            image = ['/home/jliu/research/5.MMEE/data/m2e2_rawdata/m2e2_rawdata/image/' + t for t in image]
            image = list(filter(lambda x: os.path.exists(x), image))
            image.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')
            image.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')
            image.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')

            entity = x['golden-entity-mentions']

            temp.append([sentence, event, image[:3], entity])
    return temp




def read_ace(file_name, flag=False):
    result = []
    for line in open(file_name):
        line_data = json.loads(line)
        sentences, ners, relations, events, sentence_starts, doc_keys = (line_data['sentences'], line_data['ner'], line_data['relations'], line_data['events'], line_data['sentence_start'], line_data['doc_key'])
        
        temp = []
        temp_map = []
        for ner in ners:
            ner = list(filter(lambda x: x[-1] != 'VALUE', ner))
            temp.append(ner)
            temp_map_s = set([(x[0], x[1]) for x in ner])
            temp_map.append(temp_map_s)
        ners = temp

        temp = []
        for ev, tm in zip(events, temp_map):
            # print(tm)
            temp_ev = []
            for e in ev: # each event
                temp_ev2 = []
                temp_ev2.append(e[0])
                for elem in e[1:]:
                    if (elem[0], elem[1]) in tm:
                        temp_ev2.append(elem)
                if temp_ev2:
                    temp_ev.append(temp_ev2)

            temp.append(temp_ev)
        events = temp

        for sen, ner, re, ev, sen_s in zip(sentences, ners, relations, events, sentence_starts):
            for idx in range(len(ner)):
                ner[idx][0] -= sen_s
                ner[idx][1] -= sen_s
            
            for idx in range(len(re)):
                re[idx][0] -= sen_s
                re[idx][1] -= sen_s
                re[idx][2] -= sen_s
                re[idx][3] -= sen_s

            for idx in range(len(ev)):
                ev[idx][0][0] -= sen_s
                # ev[idx][0][1] = ev[idx][0][1][ev[idx][0][1]]
                for arg in ev[idx][1:]:
                    arg[0] -= sen_s
                    arg[1] -= sen_s

        if flag:
            n = 4
            if ev and len(ev) > 3 and ev[4]:
                n = 3
            if len(ev) < 4:
                n = 3

            sentences = sentences[n:]
            ners = ners[n:]
            relations = relations[n:]
            events = events[n:]
            sentence_starts = sentence_starts[n:]

        result.append([sentences, ners, relations, events, sentence_starts, doc_keys])
    return result
    

def _to_bert_examples(words, labels):
    subword_ids = list()
    spans = list()
    label_ids = list()

    for word in words:
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        spans.append([s, e])

    for label in labels:
        label_ids.append(tag2idx.get(label, 0))

    return subword_ids, spans, label_ids


def _obtain_images(namelist):

    result = list()
    for x in all_images:
        for t in namelist:
            if t in x:
                result.append(x)
                continue
    result.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')
    result.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')
    result.append('/home/jliu/research/5.MMEE/textualEE/ACE_Image/black.jpg')

    return result[:3]



def build_bert_examples(result):
    examples = []

    for doc in result:
        sentences, ners, relations, events, sentence_starts, doc_key = doc

        images = ace_image.get(doc_key, '')
        images = _obtain_images(images)
        
        for idx, (sen, ner, re, ev, sen_s) in enumerate(zip(sentences, ners, relations, events, sentence_starts)):

            sen += ['[SEP]']
            labels = ['O'] * len(sen)
            subword_ids, spans, label_ids  = _to_bert_examples(sen, labels)

            pairs = list()
            labels = list()
            for e in ev:
                i, t = e[0]                
                t = t.split('.')[1]    ### for top1 type
                args = e[1:]

                for n in ner:
                    flag = False    
                    for a in args:
                        if n[0] == a[0]:
                            if a[-1] in config.tag2idx_role:
                                pairs.append((spans[i][0], spans[a[0]][0]))
                                labels.append(config.tag2idx_role[a[-1]])
                                flag = True

                    if not flag:
                        pairs.append((spans[i][0], spans[n[0]][0]))
                        labels.append(0)


            if pairs:
                examples.append([subword_ids, pairs, labels])

    return examples



def build_bert_examples_m2e2(result):
    examples = []

    for doc in result:
        sen, ev, image, entity = doc
        sen += ['[SEP]']
        labels = ['O'] * len(sen)
        for e in ev:
            i, t = e['trigger']['start'],  e['event_type'].split(':')[1]
            labels[i] = t

        subword_ids, spans, label_ids  = _to_bert_examples(sen, labels)
        examples.append([subword_ids, spans, label_ids, ev, entity])

    return examples


if __name__ == '__main__':

    result = read_ace('data/train.json', True)
    train = build_bert_examples(result)

    result = read_ace('data/dev.json', True)
    dev = build_bert_examples(result)

    result = read_ace('data/test.json', True)
    test = build_bert_examples(result)

    m2e2 = read_m2e2()
    m2e2 = build_bert_examples_m2e2(m2e2)
    print(len(m2e2))

    data = [train, dev, test, m2e2]
    f = open('data/data_arg.pk','wb')
    pickle.dump(data, f)