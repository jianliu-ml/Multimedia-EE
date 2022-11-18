from dataclasses import fields
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import clip
from PIL import Image
import json
import sys
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/home/jliu/data/ViT-B-32.pt", device=device)

checkpoint = torch.load("model_checkpoint/model_joint_small_1e64.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = 224
# checkpoint['model_state_dict']["context_length"] = 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

model.load_state_dict(checkpoint['model_state_dict'])

# model1, preprocess = clip.load("/home/jliu/data/ViT-B-32.pt", device=device)
# checkpoint = torch.load("model_checkpoint/model_noun_small_1e64.pt")
# model1.load_state_dict(checkpoint['model_state_dict'])

model.eval()

imsitu = json.load(open("data/imsitu_space.json"))
verbs_org = imsitu["verbs"]
verbs = [v for v in verbs_org]

roles = []
# for v in verbs:
#     roles += verbs_org[v]['order']
# roles = list(set(roles))

sr_to_ace_mapping = {}
role_to_ace_mapping = {}
for line in open('data/ace_sr_mapping.txt'):
    fields = line.strip().split()
    sr_to_ace_mapping[fields[0]] = fields[2]

    role_to_ace_mapping[fields[1]] = fields[3]

roles = list(role_to_ace_mapping.keys())

image_only = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_only_event.json').read())
image_mul = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_multimedia_event.json').read())


if __name__ == '__main__':

    image_dir = '../data/m2e2_rawdata/m2e2_rawdata/image/image'
    all_images = [image_dir + '/' + f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    all_images = sorted(all_images)
    # print(all_images[0])

    with torch.no_grad():
        text = clip.tokenize(verbs).to(device)
        text_role = clip.tokenize(roles).to(device)

        for idx, img_path in enumerate(all_images):
            print(idx, img_path)
            image_name = img_path.split('/')[-1][:-4]

            print('Gold: ', end='')
            if image_name in image_only:
                print(image_only[image_name]['event_type'], 'I')
            elif image_name in image_mul:
                print(image_mul[image_name]['event_type'], 'M')
            else:
                print('None', 'N')

            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # print(image_features.size(), text_features.size())

            logits_per_image, logits_per_text = model(image, text)
            
            logits_per_image_role, logits_per_texte_role = model(image, text_role)

            # probs = logits_per_image.cpu().numpy()[0]
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            probs_roles = logits_per_image_role.softmax(dim=-1).cpu().numpy()[0]

            xxx = (-probs).argsort()[:10]
            temp = []
            for xx in xxx:
                t = verbs[xx]
                temp.append('%s %s %f' % (t, sr_to_ace_mapping.get(t, "None"), probs[xx]))
            print('|||'.join(temp))

            xxx = (-probs_roles).argsort()[:10]
            temp = []
            for xx in xxx:
                t = roles[xx]
                temp.append('%s %s %f' % (t, role_to_ace_mapping.get(t, "None"), probs_roles[xx]))
            print('|||'.join(temp))
            print('===========')


            break

    # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
