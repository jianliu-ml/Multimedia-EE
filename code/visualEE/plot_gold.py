import json
import sys
import os
import torch
sys.path.append('Transformer_MM')
# from Transformer_MM.visualization_function import plot
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def box_corner_to_center(x1, y1, x2, y2):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = [cx, cy, w, h]
    return boxes


image_only = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_only_event.json').read())
image_mul = json.loads(open('../data/m2e2_annotations/m2e2_annotations/image_multimedia_event.json').read())

ttt = image_mul

for i_f in ttt:
    image_path = '../data/m2e2_rawdata/m2e2_rawdata/image/image/' + i_f + '.jpg'
    event_type = ttt[i_f]['event_type']
    print(event_type)
    for role in ttt[i_f]['role']:
        role_box = ttt[i_f]['role'][role]
        for i, box in enumerate(role_box):
            # box_c = box_corner_to_center(*box[1:])
            pre_fix = box[0]
            box_c = [float(x) for x in box[1:]]
            print(role, box_c)


            fig, axe = plt.subplots() 
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            w, h, _ = img.shape
            w_ratio, h_ratio = 224 / w, 224 / h

            box_c[0], box_c[1], box_c[2], box_c[3] = box_c[0] * h_ratio, box_c[1] * w_ratio, box_c[2] * h_ratio, box_c[3] * w_ratio
            img = cv2.resize(img, (224, 224))
            axe.imshow(img)

            
            axe.add_patch(patches.Rectangle(
                    (box_c[0], box_c[1]), (box_c[2] - box_c[0]), (box_c[3] - box_c[1]),
                    linewidth=1, edgecolor='r', facecolor='none'
                ))
            axe.text(box_c[0], box_c[1], role + '_' + pre_fix,
                      va='center', ha='center', fontsize=9, color='w',
                      bbox=dict(facecolor='r', lw=0))
            plt.savefig('result_gold/' + i_f + '_' + event_type + '_' + role + '_' + str(i) + '_' + pre_fix + '.jpg')