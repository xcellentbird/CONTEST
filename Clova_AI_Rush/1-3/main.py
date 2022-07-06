import os
import math
import datetime

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.nn.functional as F

from data_local_loader import data_loader

from tqdm import tqdm

import timm
from pprint import pprint
from utils import AverageMeter
from ptflops import get_model_complexity_info
from torch.autograd import Variable

try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
except:
    IS_ON_NSML = False
    DATASET_PATH = '../1-3-DATA-fin'

#test
LABELS = [['90', '82', '85', '93', '65', '57', '100', '0', '69', '39', '23'],
          ['51', '6', '104', '4', '1', '105', '27', '103', '44', '95', '70', '78', '28', '40', '3', '96', '17', '91', '84', '11', '106', '99', '5', '54', '24', '47', '52', '74', '67', '29', '63', '58', '88', '60', '53', '35', '2', '49', '26', '50', '46', '68', '102', '66', '89', '-1'],
          ['30', '19', '61', '13', '59', '76', '15', '48', '81', '18', '16', '31', '75', '45', '41', '55', '14', '64', '33', '10', '71', '42', '20', '43', '56', '77', '72', '73', '34', '38', '80', '79', '-1']]

HIERARCHY = [{69: [70, 74, 78], 23: [35, 24, 25, 26, 27, 28, 29], 39: [40, 44, 46, 47, 49, 50, 51, 52, 53, 54], 0: [1, 2, 3, 4, 5, 6, 11, 17, 88], 100: [101, 102, 103, 104, 105, 106], 65: [66, 67, 68], 93: [96, 97, 98, 99, 94, 95], 90: [91, 92], 57: [58, 60, 63], 82: [89, 83, 84, 87], 85: []},
             {78: [80, 81, 79], 35: [36, 37, 38], 54: [56, 55], 17: [18, 19, 20, 21, 22], 40: [41, 42, 43], 29: [32, 33, 34, 30, 31], 46: [], 11: [12, 13, 14, 15, 16], 106: [], 49: [], 67: [], 28: [], 103: [], 70: [72, 73, 71], 99: [], 88: [], 44: [45], 91: [], 53: [], 58: [59], 51: [], 74: [75, 76, 77], 5: [], 84: [], 47: [48], 50: [], 63: [64], 96: [], 2: [], 105: [], 24: [], 60: [61, 62], 52: [], 102: [], 27: [], 26: [], 6: [8, 9, 10, 7], 68: [], 104: [], 4: [], 95: [], 3: [], 66: [], 1: [], 89: [], 83: [], 101: [], 98: [], 87: [], 94: [], 25: [], 92: [], 97: []}]

INT_LABELS2 = [[69, 23, 39, 0, 100, 65, 93, 90, 57, 82, 85],
               [78, 35, 54, 17, 40, 29, 46, 11, 106, 49, 67, 28, 103, 70, 99, 88, 44, 91, 53, 58, 51, 74, 5, 84, 47, 50, 63, 96, 2, -1, 105, 24, 60, 52, 102, 27, 26, 6, 68, 104, 4, 95, 3, 66, 1, 89, 83, 101, 98, 87, 94, 25, 92, 97],
               [81, 38, 56, 19, 42, 30, -1, 14, 31, 16, 34, 43, 45, 59, 13, 77, 76, 48, 55, 64, 71, 73, 20, 75, 79, 61, 72, 33, 15, 41, 10, 18, 80, 12, 22, 8, 32, 37, 21, 62, 7, 36, 9]]


def get_num_label(batch_labels, follow_hierarchy = False):
    # (batch_size, num_classes) 크기의 torch.tensor 입력이 들어온다.
    HIERARCHY = [{69: [70, 74, 78], 23: [35, 24, 25, 26, 27, 28, 29], 39: [40, 44, 46, 47, 49, 50, 51, 52, 53, 54],
                  0: [1, 2, 3, 4, 5, 6, 11, 17, 88], 100: [101, 102, 103, 104, 105, 106], 65: [66, 67, 68],
                  93: [96, 97, 98, 99, 94, 95], 90: [91, 92], 57: [58, 60, 63], 82: [89, 83, 84, 87], 85: []},
                 {78: [80, 81, 79], 35: [36, 37, 38], 54: [56, 55], 17: [18, 19, 20, 21, 22], 40: [41, 42, 43],
                  29: [32, 33, 34, 30, 31], 46: [], 11: [12, 13, 14, 15, 16], 106: [], 49: [], 67: [], 28: [], 103: [],
                  70: [72, 73, 71], 99: [], 88: [], 44: [45], 91: [], 53: [], 58: [59], 51: [], 74: [75, 76, 77], 5: [],
                  84: [], 47: [48], 50: [], 63: [64], 96: [], 2: [], 105: [], 24: [], 60: [61, 62], 52: [], 102: [],
                  27: [], 26: [], 6: [8, 9, 10, 7], 68: [], 104: [], 4: [], 95: [], 3: [], 66: [], 1: [], 89: [],
                  83: [], 101: [], 98: [], 87: [], 94: [], 25: [], 92: [], 97: []}]
    INT_LABELS2 = [[69, 23, 39, 0, 100, 65, 93, 90, 57, 82, 85],
                   [78, 35, 54, 17, 40, 29, 46, 11, 106, 49, 67, 28, 103, 70, 99, 88, 44, 91, 53, 58, 51, 74, 5, 84, 47,
                    50, 63, 96, 2, -1, 105, 24, 60, 52, 102, 27, 26, 6, 68, 104, 4, 95, 3, 66, 1, 89, 83, 101, 98, 87,
                    94, 25, 92, 97],
                   [81, 38, 56, 19, 42, 30, -1, 14, 31, 16, 34, 43, 45, 59, 13, 77, 76, 48, 55, 64, 71, 73, 20, 75, 79,
                    61, 72, 33, 15, 41, 10, 18, 80, 12, 22, 8, 32, 37, 21, 62, 7, 36, 9]]

    ret_labels = torch.tensor([[0, 0, 0] for _ in range(len(batch_labels[0]))], dtype=torch.int8)  # 반환값은 batch_size * 3 크기의 int형 label

    for i, layer in enumerate(batch_labels):
        for j, probs in enumerate(layer):
            if i > 0:
                probs = probs.clone().detach()
                probs[INT_LABELS2[i].index(-1)] = 0.0
            leaf = []
            if i > 0:
                root = ret_labels[j][i-1]
                if root != -1:
                    leaf = HIERARCHY[i-1][int(root)]

            if leaf and follow_hierarchy:
                z = torch.Tensor([0. for _ in probs])
                for k in leaf:
                    idx = INT_LABELS2[i].index(k)
                    z[idx] = probs[idx]
                max_prob = torch.argmax(z)
            else:
                max_prob = torch.argmax(probs)
            ret_labels[j][i] = int(INT_LABELS2[i][max_prob])
    """
    for pred_label in batch_labels:
        ret_label = [-1, -1, -1]
        for i, layer in enumerate(hierarchy):
            cmp_map = {idx: pred_label[idx] for idx in layer}
            if follow_hierarchy and i:
                tmp = {idx: pred_label[idx] for idx in HIERARCHY[i-1][ret_label[i-1]]}
                if tmp:
                    cmp_map = tmp

            ret_label[i] = max(cmp_map, key=lambda idx: cmp_map[idx])
        ret_labels.append(ret_label)
    """
    return ret_labels


def reputation(pred_labels, goal_labels, root_only=3):
    # (batch_size * 3) 크기의 str 형태 라벨들을 받아온다.
    scores = 0.0
    for c in range(len(pred_labels)):
        score = 0.0
        denom = 0.0
        for i, (goal, pred) in enumerate(zip(goal_labels[c][:root_only], pred_labels[c][:root_only])):
            goal = INT_LABELS2[i][goal]
            if goal > -1:
                denom += 1.0
                if goal == pred:
                    score += 1.0
            else:
                break

        scores += score / denom
    return scores


def _infer(model, root_path, test_loader=None):

    if test_loader is None:
        test_loader = data_loader(root=root_path, phase='test', split=0.0, batch_size=1, submit=True)

    model.eval()
    ret_id = []
    ret_cls = []
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        data_id = data_id[0].item()
        res_id = data_id

        fc = model(image)
        # fc = fc.squeeze().detach().cpu().numpy()
        res_cls = get_num_label(fc, follow_hierarchy=True)[0].detach()
        res_cls = res_cls.cpu().numpy()
        # res_cls = [-1, -1, -1]  # np.argmax(fc)

        # int형이 아닌 str형 보내는 것도 시도해볼 것.
        ret_cls.append(res_cls)
        ret_id.append(res_id)
    print(ret_cls)
    return [ret_id, ret_cls]


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


num_classes = [11, 54, 43]


class DnnClassifier(nn.Module):
    def __init__(self, in_features=1536, num_classes=num_classes):
        super(DnnClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)
        self.classifier1 = nn.Linear(in_features=512, out_features=num_classes[0])
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(in_features=in_features, out_features=512)
        self.classifier2 = nn.Linear(in_features=512, out_features=num_classes[1])
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(in_features=in_features, out_features=512)
        self.bn3 = nn.BatchNorm1d(512)
        self.classifier3 = nn.Linear(in_features=512, out_features=num_classes[2])

    def forward(self, x):
        x3 = F.silu(self.bn3(self.fc3(x)))
        out3 = self.classifier3(x3)
        out3 = torch.sigmoid(out3)

        x2 = F.silu(self.bn2(self.fc2(x)))
        out2 = self.classifier2(x2 + x3)
        out2 = torch.sigmoid(out2)

        x1 = F.silu(self.bn1(self.fc1(x)))
        out1 = self.classifier1(x1 + x2 + x3)
        out1 = torch.sigmoid(out1)

        return [out1, out2, out3]


class MySuperUltraUniverseFCN(nn.Module):
    def __init__(self):
        super(MySuperUltraUniverseFCN, self).__init__()
        self.out_features = 8
        self.conv1 = nn.Conv2d(3, self.out_features, kernel_size=224)  # 4 * 16
        self.bn1 = nn.BatchNorm2d(self.out_features)

    def forward(self, out):
        out = F.silu(self.bn1(self.conv1(out)))

        return out


class EnsembleModel(nn.Module):
    def __init__(self, out_features=1000, num_classes=num_classes):
        super(EnsembleModel, self).__init__()
        self.out_features = 0
        self.modelA = timm.create_model('tf_efficientnetv2_b3', pretrained=True)
        self.out_features += self.modelA.classifier.in_features
        self.modelB = timm.create_model('rexnet_100', pretrained=True)
        self.out_features += self.modelB.head.fc.in_features
        self.modelC = timm.create_model('tf_mobilenetv3_small_minimal_100', pretrained=True)
        self.out_features += self.modelC.classifier.in_features
        self.modelA.classifier = nn.Identity()
        self.modelB.head.fc = nn.Identity()
        self.modelC.classifier = nn.Identity()

        self.modelFCN = MySuperUltraUniverseFCN()

        self.classifier = DnnClassifier(in_features=self.out_features + self.modelFCN.out_features)
        # self.classifier2 = DnnClassifer2(in_features=self.out_features + self.modelFCN.out_features)


    def forward(self, out):
        out1 = self.modelA(out.clone())
        out1 = out1.view(out1.size(0), -1)

        out2 = self.modelB(out.clone())
        out2 = out2.view(out2.size(0), -1)

        out3 = self.modelC(out.clone())
        out3 = out3.view(out3.size(0), -1)

        out4 = self.modelFCN(out)
        out4 = out4.view(out4.size(0), -1)

        out_teacher = torch.cat((out1, out2, out3, out4), dim=1)
        #out_student = torch.cat((out1, out2, out3, out4), dim=1)

        out_teacher = self.classifier(F.silu(out_teacher))
        #out_student = self.classifier2(F.silu(out_student))
        return out_teacher#, out_student


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(input, target):
    input = input.clone().detach()
    target = target.clone().detach()
    beta = 1.0
    cutmix_prob = 1.0
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return target_b, lam


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=107)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--train_split", type=float, default=0.99315)  # default=0.99315)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=3000)
    args.add_argument("--print_iter", type=int, default=10)

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    # get configurations
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    mode = config.mode
    train_split = config.train_split
    batch_size = config.batch_size

    """----------note---------"""
    print('------------ last: ensemble 미적용 -----------')
    print('model: efficientnet v2 b3')
    print('color jitter image augmentation 적용')
    print('CELoss에 weight 적용 for Unbalanced Data')
    print('cutmix 적용 balanced weight와 혼용이 될 지 모르겠다...')
    print('아무래도 cutmix가 ignore index를 만날 때 큰 반감이 일어나는 것 같다.')
    print('제일 중요한 첫번째 계층에 적용하고 후반부 epoch에서 나머지 계층에도 적용해야겠다')
    print('두, 세번째 계층에 강한 weight 비율 적용 - 이건 좀 많이 아닌듯')
    print('model ensemble rexnet 100 추가 적용')
    print('ensemble cat 적용')
    print('tf_mobilenetv3_small_minimal_100 ensemble 적용')
    print('각 모델의 fc, classifier를 identity로 고정 후 cat')
    print('color jitter 미적용 - 음식의 색깔이 바뀌면 감별하기 어려워질 듯')
    print('batchnormaliz가 있기 때문에 lr을 두배로 높임 - 학습 속도 향상')
    print('color jitter에 밝기만 0.2 적용 사진 밝기')
    print('그냥 color jitter 없앰... 이게 더 효과 좋아서')
    print('random perspective도 없애보자. 사진의 대부분의 각도은 정해져있을 것 같다.')
    print('random perspective 없애면 안 됏..')
    print('loss에 가중치 1.1을 곱하였다')
    print('cutmix를 epoch에 상관없이 바로 적용시켰다.')
    print('loss에 비율 1이 넘도록 가중치를 곱하면 안 된다. 왤까')
    print('DNN 층을 깊게 했더니 성능이 더 안 좋아졌다. 더 깊게 할 순 없을까?')
    print('장난질: dnn층을 최소화하기 위해 1번째 라벨, 2번째, 3번째 분류기를 바로바로 이었다.')
    print('결과는 역시 Badd. 원래대로 되돌려 놓고 다시 해보자')
    print('cutmix를 바로 적용하면 좋지 않았다. perspective를 없애고 해보자')
    print('그냥 좋지 않음... 이번엔 transforms 위치를 바꿔보자. perspective 추가')
    print('똑같다.')
    print('모델에서 세번째 계층 label을 제일 앞에 두니 세번째 계층의 성적이 좋았다. 첫번째는 워낙 데이터가 많아서 좋고')
    print('또한가지, -1 라벨을 무시하게 되면 cutmix과정에서 loss가 낮게 잡히는 것 같다. = 학습이 잘 되지 않는다')
    print('-1 데이터가 많긴 하지만, 이는 class weight가 carry 해준다.')
    print('이번엔 모델에서 각 계층이 독립적으로 학습되도록 만들어보자. 첫계층은 특징이 강하고 데이터가 많으니 잘 나올 것이고')
    print('이는 마지막 평가에서 어차피 계층별로 라벨을 재 분류하기 때문에 더 좋게 나올 것 같다.')
    print('결과는 오히려 마지막 재분류할 때 좋지 않은 결과가 나온다. 서로 연관성이 없으니 마지막에 싸우는 듯...')
    print('이번엔 입력에 그 전 라벨의 fc 결과값을 더해줌으로서 연결시켜보자')
    print('효과는 좋지만, 이게 언더피팅이 일어나는 것 같다. lr을 1/5로 줄여보자')
    print('언더피팅. lr을 1/10(1e-4)로 줄여보자')
    print('아주 매애애우 좋은 성과가 나왔다. 언더피팅이지만. lr을 더 줄여봐야겠다.')
    print('모델 부에서 이미 계층을 이어놓은 상태다. 결과값에서 계층을 잇는다면 오히려 결과가 안 좋다.')
    print('lr를 1/100으로 줄이자. 그리고 괜찮으면 valid 데이터를 동원해서 학습시켜보자')
    print('훨씬 안 좋게 나왔다. 이렇게 lr영향이 클 줄은...')
    print('생각해보니 스케쥴러가 이미 lr을 관리해주고 있다. 줄여줄 필요가 없다. 다시 1e-4로')
    print('5e-5 lr로 설정')
    print('colorjitter에서 밝기 채도 0.2 설정 - 아이폰과 삼성 카메라 사진 차이. lr 0.0002설정. cutmix 5lr에서 적용')
    print('colorjitter 0.1씩 수정. lr0.0001. cutmix 2 lr부터. 1, 2계층에 dropout적용(0.1씩) 언더피팅을 유도')
    print('dropout 효과가 좋다.  0.5로 첫 층만')
    print('argnum에서 -1인 부분은 빼고 판정하도록 설정')
    print('test셋에 novel셋이 있는 것 같다. 아마 2, 3계층의 것이 아닐까 싶다. 그렇다면 follow hierarchy를 적용하자')
    print('일단 efficientnet ns 모델을 가져와서 실험해보자. 안 되면 직접 만들어야지')
    print('fcn 만들어 ensemble')
    # initialize model using timm
    # pprint(timm.list_models(pretrained=True))
    # print('dataset path: ', DATASET_PATH)
    model = EnsembleModel()
    # model = timm.create_model('tf_efficientnetv2_b3', pretrained=True)
    # model = EnsembleModel(out_features=1024)
    # out_features = model.classifier.in_features
    # model.classifier = DnnClassifier()

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # init_weight(model)

    if cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=base_lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())

    if mode == 'train':
        tr_loader = data_loader(root=DATASET_PATH, phase='train', split=train_split, batch_size=batch_size, submit=False)
        valid_loader = data_loader(root=DATASET_PATH, phase='test', split=train_split, batch_size=batch_size, submit=False)

        time_ = datetime.datetime.now()
        num_batches = len(tr_loader)
        
        train_stat = AverageMeter()
        print("train data size:", len(tr_loader.dataset))
        valid_dataset_size = len(valid_loader.dataset)
        print("valid data size:", valid_dataset_size)

        global_iter = 0
        for epoch in range(num_epochs):

            model.train()
            for iter_, data in enumerate(tr_loader):
                global_iter += iter_

                _, x, label_0, label_1, label_2, class_weights = data

                loss = torch.autograd.Variable(torch.FloatTensor(1)).zero_()
                loss_fn_0 = nn.CrossEntropyLoss(weight=class_weights[0][0])
                loss_fn_1 = nn.CrossEntropyLoss(weight=class_weights[1][0]) #, ignore_index=INT_LABELS2[1].index(-1))
                loss_fn_2 = nn.CrossEntropyLoss(weight=class_weights[2][0]) #, ignore_index=INT_LABELS2[2].index(-1))

                if cuda:
                    x = x.cuda()
                    label_0 = label_0.cuda().squeeze(1)
                    label_1 = label_1.cuda().squeeze(1)
                    label_2 = label_2.cuda().squeeze(1)
                    loss = loss.squeeze().cuda()

                    loss_fn_0 = loss_fn_0.cuda()
                    loss_fn_1 = loss_fn_1.cuda()
                    loss_fn_2 = loss_fn_2.cuda()

                pred = model(x)

                if epoch >= 10.0:
                    # CutMix
                    label_0b, lam_0 = cutmix(x, label_0)
                    label_1b, lam_1 = cutmix(x, label_1)
                    label_2b, lam_2 = cutmix(x, label_2)
                    loss1 = loss_fn_0(pred[0], label_0) * lam_0 + loss_fn_0(pred[0], label_0b) * (1. - lam_0)
                    loss2 = loss_fn_1(pred[1], label_1) * lam_1 + loss_fn_1(pred[1], label_1b) * (1. - lam_1)
                    loss3 = loss_fn_2(pred[2], label_2) * lam_2 + loss_fn_2(pred[2], label_2b) * (1. - lam_2)
                else:
                    loss1 = loss_fn_0(pred[0], label_0)
                    loss2 = loss_fn_1(pred[1], label_1)
                    loss3 = loss_fn_2(pred[2], label_2)
                """
                loss1 = loss_fn_0(pred[0], label_0)
                loss2 = loss_fn_1(pred[1], label_1)
                loss3 = loss_fn_2(pred[2], label_2)
                """
                # very naive loss function given
                loss += loss1 + loss2 + loss3

                train_stat.update(loss, x.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({:.5f} + {:.5f} + {:.5f} = {:.5f}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss1, loss2, loss3, train_stat.avg, elapsed, expected))
                    time_ = datetime.datetime.now()

                    if IS_ON_NSML:
                        report_dict = dict()
                        report_dict["train__loss"] = float(train_stat.avg)
                        report_dict["train__lr"] = optimizer.param_groups[0]["lr"]
                        nsml.report(step=global_iter, **report_dict)

            scheduler.step()

            if IS_ON_NSML:
                nsml.save(str(epoch + 1))

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

            if valid_dataset_size != 0:
                model.eval()
                total_score = 0
                first_root_score = 0
                second_root_score = 0
                valid_loss = torch.autograd.Variable(torch.FloatTensor(1)).zero_()
                score_follow_hierarchy = [0, 0, 0]
                with torch.no_grad():
                    for iter_, data in enumerate(valid_loader):
                        _, x, label_0, label_1, label_2 = data

                        if cuda:
                            x = x.cuda()
                            label_0 = label_0.cuda()
                            label_1 = label_1.cuda()
                            label_2 = label_2.cuda()
                            valid_loss = valid_loss.cuda()

                        pred = model(x)

                        valid_loss1 = loss_fn_0(pred[0], label_0.squeeze(1))
                        valid_loss2 = loss_fn_1(pred[1], label_1.squeeze(1))
                        valid_loss3 = loss_fn_2(pred[2], label_2.squeeze(1))
                        valid_loss += valid_loss1 + valid_loss2 + valid_loss3

                        pred_label = get_num_label(pred)
                        follow_pred = get_num_label(pred, follow_hierarchy=True)
                        goal_label = torch.cat((label_0, label_1, label_2), 1)
                        total_score += reputation(pred_labels=pred_label, goal_labels=goal_label)
                        first_root_score += reputation(pred_labels=pred_label, goal_labels=goal_label, root_only=1)
                        second_root_score += reputation(pred_labels=pred_label, goal_labels=goal_label, root_only=2)
                        for i in range(3):
                            score_follow_hierarchy[i] += reputation(pred_labels=follow_pred, goal_labels=goal_label, root_only=i+1)

                per_score = (100. / valid_dataset_size) * total_score
                first_root_score = (100. / valid_dataset_size) * first_root_score
                second_root_score = (100. / valid_dataset_size) * second_root_score
                print('[epoch {}] test loss: {:.5f} + {:.5f} + {:.5f} = {:.5f}, test score: {} - {} - {} '.format(epoch + 1, float(valid_loss1), float(valid_loss2), float(valid_loss3), float(valid_loss), first_root_score, second_root_score, per_score))

                for i in range(3):
                    score_follow_hierarchy[i] = (100. / valid_dataset_size) * score_follow_hierarchy[i]
                print('                                           follow hierarchy score: {} - {} - {} '.format(score_follow_hierarchy[0], score_follow_hierarchy[1], score_follow_hierarchy[2]))