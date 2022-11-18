import torch
import pickle

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def _to_file(data, filename):
    f = open(filename, 'w')
    for elem in data:
        words = ' '.join(elem[1])
        print(words, file=f)
    f.close()

# with open('data/EE/data.pickle', 'rb') as f:
#     data = pickle.load(f)

# train, dev, test = data['train'], data['val'], data['test']
# print(len(train), len(dev), len(test))
# print(train[0])

# _to_file(train, 'train.txt')
# _to_file(dev, 'dev.txt')
# _to_file(test, 'test.txt')


if __name__ == '__main__':
    with open('data/EE/data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    train, dev, test = data['train'], data['val'], data['test']
    print(len(train), len(dev), len(test))
    print(train[0])

    _to_file(train, 'train.txt')
    _to_file(dev, 'dev.txt')
    _to_file(test, 'test.txt')




