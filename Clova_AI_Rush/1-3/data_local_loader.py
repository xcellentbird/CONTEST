from torch.utils import data
from torchvision import datasets, transforms
import torch
import os
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

"""
labels
[{'90', '82', '85', '93', '65', '57', '100', '0', '69', '39', '23'}, 
{'51', '6', '104', '4', '1', '105', '27', '103', '44', '95', '70', '78', '28', '40', '3', '96', '17', '91', '84', '11', '106', '99', '5', '54', '107', '24', '47', '52', '74', '67', '29', '63', '58', '88', '60', '53', '35', '2', '49', '26', '50', '46', '68', '102', '66', '89'}, 
{'30', '19', '61', '13', '59', '76', '15', '48', '81', '18', '16', '31', '75', '107', '45', '41', '55', '14', '64', '33', '10', '71', '42', '20', '43', '56', '77', '72', '73', '34', '38', '80', '79'}]

"""
INT_LABELS3 = [[69, 23, 39, 0, 100, 65, 93, 90, 57, 82, 85],
               [78, 35, 54, 17, 40, 29, 46, 11, 106, 49, 67, 28, 103, 70, 99, 88, 44, 91, 53, 58, 51, 74, 5, 84, 47, 50, 63, 96, 2, -1, 105, 24, 60, 52, 102, 27, 26, 6, 68, 104, 4, 95, 3, 66, 1, 89, 83, 101, 98, 87, 94, 25, 92, 97],
               [81, 38, 56, 19, 42, 30, -1, 14, 31, 16, 34, 43, 45, 59, 13, 77, 76, 48, 55, 64, 71, 73, 20, 75, 79, 61, 72, 33, 15, 41, 10, 18, 80, 12, 22, 8, 32, 37, 21, 62, 7, 36, 9]]
INT_LABELS2 = [[], [], []]
edges = [dict(), dict()]
num_classes = 107

def pil_loader(path, img_size=224):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(e)


def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ColorJitter(brightness=0.1, saturation=0.1))
        # transform.append(transforms.RandomPerspective())
        transform.append(transforms.RandomResizedCrop(224))
    else:
        transform.append(transforms.CenterCrop(224))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)


class TestDataset(data.Dataset):
    def __init__(self, root='../1-3-DATA-fin'):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'test'
        self.data_loc = 'test_data'
        self.path = os.path.join(root, self.data_loc, self.data_idx)

        with open(self.path, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            idx = line.split("_")[0]
            self.samples.append([line.rstrip('\n'), idx])

        self.transform = get_transform(random_crop=False)

    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, idx = self.samples[index]
        path = os.path.join(self.root, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        return torch.LongTensor([int(idx)]), sample

    def __len__(self):
        return len(self.samples)


class CustomDataset(data.Dataset):
    def __init__(self, is_train=True, root='../1-3-DATA-fin', split=1.0):

        self.root = root
        self.data_idx = 'data_idx'
        self.is_train = is_train
        self.sample_dir = 'train'
        self.data_loc = 'train_data'
        self.path = os.path.join(root, self.sample_dir, self.data_loc, self.data_idx)

        with open(self.path , 'r') as f:
            lines = f.readlines()

        test_size = 1.0 - split
        if test_size > 0:
            train_lines, valid_lines = train_test_split(lines, test_size= 1.0 - split, random_state=97)
        else:
            train_lines = lines
            valid_lines = []

        # split = int(len(lines) * split)
        if is_train:
            random_crop = True
            lines = train_lines  # lines[:split]
        else:
            random_crop = False
            lines = valid_lines  # lines[split:]

        self.samples = []
        self.labels = [[], [], []]
        global edges
        for cnt_flg, line in enumerate(lines):
            idx = line.split(" ")[0].split("__")[1].split("_")[0]
            label = [v.rstrip('\n') for v in line.split(' ')[1:]]

            # 라벨의 계층이 3칸 미만일 경우 '-1'을 덧붙인다. -1은 공백으로 평가한다.
            if len(label) < 3:
                label = (label + ['-1', '-1'])[:3]

            # test for 1st hierarchy layer
            # label = [label[0]]

            label = list(map(int, label))

            # 계층별 라벨 id 분류
            for i, lb in enumerate(label[:2]):
                if lb == -1:
                    break
                if lb not in edges[i]:
                    edges[i][lb] = set()
                if label[i+1] != -1:
                    edges[i][lb].add(label[i + 1])

            for i in range(len(label)):
                label[i] = INT_LABELS3[i].index(label[i])
                self.labels[i].append(label[i])

            self.samples.append([line.split(' ')[0], label, idx])

        if is_train:
            self.labels = np.array(self.labels)
            self.class_weights = [torch.Tensor(compute_class_weight(class_weight='balanced', classes=np.unique(self.labels[i]), y=self.labels[i])) for i in range(3)]

        self.transform = get_transform(random_crop=random_crop)
        # print(len(INT_LABELS2[0]), len(INT_LABELS2[1]), len(INT_LABELS2[2]))

    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        # target_idx = [0.0] * num_classes
        # for t in target:
        #     target_idx[t] = 1.0

        if self.is_train:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])]), self.class_weights
        return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])])

    def __len__(self):
        return len(self.samples)


def data_loader(root, phase='train', batch_size=16, split=1.0, submit=True):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    if submit:
        dataset = TestDataset(root=root)
    else:
        dataset = CustomDataset(is_train=is_train, root=root, split=split)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)


