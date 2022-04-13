# 数据预处理
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

'''
an example of img_items:
( '0709_17.png',
  'Alphabet_of_the_Magi/character01',
  './../datasets/omniglot/python/images_background/Alphabet_of_the_Magi/character01')
'''

root_dir = '/home/jixi-li/WORK/Research/dataset/Few shot/omniglot_standard/python'
root_dir_train = os.path.join(root_dir, 'images_background')
root_dir_test = os.path.join(root_dir, 'images_evaluation')

def find_classes(root_dir_train):
    img_items = []
    for (root, dirs, files) in os.walk(root_dir_train):
        for file in files:
            if (file.endswith("png")):
                r = root.split('/')
                img_items.append((file, r[-2] + "/" + r[-1], root))
    print("== Found %d items " % len(img_items))
    return img_items

## 构建一个词典{class:idx}
def index_classes(items):
    class_idx = {}
    count = 0
    for item in items:
        if item[1] not in class_idx:
            class_idx[item[1]] = count
            count += 1
    print('== Found {} classes'.format(len(class_idx)))
    return class_idx


img_items_train =  find_classes(root_dir_train) # [(file1, label1, root1),..]
img_items_test = find_classes(root_dir_test)

class_idx_train = index_classes(img_items_train)
class_idx_test = index_classes(img_items_test)


def generate_temp(img_items,class_idx):
    temp = dict()
    for imgname, classes, dirs in img_items:
        img = '{}/{}'.format(dirs, imgname)
        label = class_idx[classes]
        transform = transforms.Compose([lambda img: Image.open(img).convert('L'),
                                  lambda img: img.resize((28,28)),
                                  lambda img: np.reshape(img, (28,28,1)),
                                  lambda img: np.transpose(img, [2,0,1]),
                                  lambda img: img/255.
                                  ])
        img = transform(img)
        if label in temp.keys():
            temp[label].append(img)
        else:
            temp[label] = [img]
    print('begin to generate omniglot.npy')
    return temp
    ## 每个字符包含20个样本

temp_train = generate_temp(img_items_train, class_idx_train)
temp_test = generate_temp(img_items_test, class_idx_test)

img_list = []
for label, imgs in temp_train.items():
    img_list.append(np.array(imgs))
img_list = np.array(img_list).astype(np.float) # [[20 imgs],..., 1623 classes in total]
print('data shape:{}'.format(img_list.shape)) # (964, 20, 1, 28, 28)
np.save(os.path.join(root_dir, 'omniglot_train.npy'), img_list)
print('end.')


img_list = []
for label, imgs in temp_test.items():
    img_list.append(np.array(imgs))
img_list = np.array(img_list).astype(np.float) # [[20 imgs],..., 1623 classes in total]
print('data shape:{}'.format(img_list.shape)) # (659, 20, 1, 28, 28)

np.save(os.path.join(root_dir, 'omniglot_test.npy'), img_list)
print('end.')