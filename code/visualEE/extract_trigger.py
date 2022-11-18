import json

image_only = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_only_event.json').read())
image_mul = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_multimedia_event.json').read())

mapping = {}
for line in open('data/ace_sr_mapping.txt'):
    fields = line.strip().split()
    mapping[(fields[0], fields[1])] = fields[3]

sr_to_ace_mapping = {}
for line in open('data/ace_sr_mapping.txt'):
    fields = line.strip().split()
    sr_to_ace_mapping[fields[0]] = fields[2].replace('||', ':').replace('|', '-')

ace_roles = {}
for line in open('data/ace_sr_mapping.txt'):
    fields = line.strip().split()
    key = fields[2].replace('||', ':').replace('|', '-')
    ace_roles.setdefault(key, set())
    ace_roles[key].add((fields[1], fields[3]))

imsitu = json.load(open("data/imsitu_space.json"))
verbs = imsitu["verbs"]
nouns = imsitu['nouns']

gold = []

lines = open('zero_result.txt').readlines()

img_files = lines[0::5]
img_files = [x.split(' ')[1].split('/')[-1].strip()[:-4] for x in img_files]

# print(img_files)

gold_org = lines[1::5]
gold = [x.strip().split()[1] for x in gold_org]
data_split = [x.strip().split()[2] for x in gold_org]

pred = lines[2::5]
pred_roles = lines[3::5]

def _find_answer(predictions, probs):
    r = {'None': 0}
    r_count = {'None': 0}
    for pred, prob in zip(predictions, probs):
        if pred == 'None':
            continue
        r.setdefault(pred, 0)
        r[pred] += prob

        r_count.setdefault(pred, 0)
        r_count[pred] += 1
    
    first_key = 'xxx'
    first_score = 100

    for t, p in zip(predictions, probs):
        if t != 'None':
            first_score = p
            first_key = t
            break

    k = max(r, key=r.get)
    if k != 'None' and (r[k] > probs[0] or r_count[k] > len(probs) / 3):
        ans = k
    else:
        ans = 'None'
    
    if k == predictions[0] and probs[0] > 0.57:
        ans = k
    
    if probs[0] < 0.091:
        ans = 'None'

    return ans

set_pred = 0
set_correct = 0
set_gold = 0

predictions = []
for i_f, g, p, p_r in zip(img_files, gold, pred, pred_roles):

    xxx = p.strip().split('|||')[:]

    my_list = ['destroying', 'saluting', 'subduing', 'gathering', 'ejecting', 'marching', 'aiming', 'confronting', 'bulldozing']
    for m in my_list:
        sr_to_ace_mapping[m] = 'None'

        for l in mapping:
            if l[0] == m:
                mapping[l] = None
    
    # aaa = ['floating', 'leading', 'cheering', 'restraining', 'bulldozing', 'mourning', 'tugging']
    # bbb = ['Movement:Transport', 'Contact:Meet', 'Conflict:Demonstrate', 'Justice:Arrest-Jail', 'Conflict:Attack', 'Life:Die', 'Conflict:Attack']

    aaa = {
        "floating" : "Movement:Transport",
        "leading" : "Contact:Meet",
        "cheering" : "Conflict:Demonstrate",
        "restraining" : "Justice:Arrest-Jail",
        "bulldozing" : "Conflict:Attack",
        "mourning" : "Life:Die",
        "tugging" : "Conflict:Attack",
        'signing': 'Contact:Meet',
        'colliding': "Conflict:Attack", 
        'weighing': "Movement:Transport",
        'sleeping': "Life:Die",
        'falling': "Life:Die",
        'confronting': 'Contact:Meet',
        'gambling': 'Transaction:Transfer-Money',
        'pricking': 'Transaction:Transfer-Money'
    }

    for a in aaa:
        b = aaa[a]
        sr_to_ace_mapping[a] = b

        for c in ace_roles[b]:
            if (a, c[0]) not in mapping:
                mapping[(a, c[0])] = c[1] 

    x = [sr_to_ace_mapping.get(x.split()[0], 'None').replace('||', ':').replace('|', '-') for x in xxx]
    p = [float(x.split()[2]) for x in xxx]

    ans = _find_answer(x, p)
    predictions.append(ans)

    # if g == ans and g != 'None':
    #     print(i_f)
    #     print(g)
    #     print(x)
    #     print([x.split()[0] for x in xxx])
    #     print(p)
    #     print(ans)
    #     print('=========================')

    org_names = [x.split()[0] for x in xxx]
    org_names = list(filter(lambda x: sr_to_ace_mapping.get(x, 'XXX') == ans, org_names))

    xxx = p_r.strip().split('|||')

    p = None
    if i_f in image_only:
        p = image_only[i_f]['role']
    elif i_f in image_mul:
        p = image_mul[i_f]['role']
    else:
        p = list()

    temp_g = list()
    for r in p:
        temp_g += [r] * len(p[r])

    res_role_for_plot = list()

    temp_p = list()
    c = {}
    for x in xxx:
        r, _, v = x.split()
        for o in org_names:
            z = mapping.get((o, r))
            if z:
                res_role_for_plot.append((r, z))
                c.setdefault(z, 0)
                c[z] += float(v)
                temp_p.append(z)
    
    temp_p = list(c.keys())
    temp_g = list(set(temp_g))

    print(i_f)
    print(ans)
    print(list(set(res_role_for_plot)))
    print('=====')

    set_gold += len(temp_g)
    set_pred += len(temp_p)

    from collections import Counter
    c = list((Counter(temp_g) & Counter(temp_p)).elements())

    set_correct += len(c)

# print(gold)
# print(pred)

from sklearn.metrics import precision_recall_fscore_support
print('Event Extraction:')
print(precision_recall_fscore_support(gold, predictions, labels=['Transaction:Transfer-Money', 'Movement:Transport', 'Conflict:Attack', 'Contact:Meet', 'Justice:Arrest-Jail', 'Life:Die', 'Conflict:Demonstrate', 'Contact:Phone-Write'], average='micro'))


print('Argument Extraction:')

p = set_correct / set_pred
r = set_correct / set_gold
f1 = 2 * p * r / (p + r)

print(p, r, f1)