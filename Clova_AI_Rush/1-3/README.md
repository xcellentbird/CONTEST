# 1-3 네이버 클라우드 마이박스의 이미지 분류

네이버 클라우드 마이박스의 이미지를 테마 별로 분리합니다.

# Introduction

이 문제는 실제 마이박스에서 수집된 이미지 테마들을 분류하는 문제입니다. 이 문제의 특성은 아래와 같습니다.

(1) multi-label classification 문제입니다.

(2) 각 이미지 별로 multi-class 형태의 label들은 hierarchy를 띄고 있습니다.

![Figure](https://media.oss.navercorp.com/user/8335/files/06e55980-ae76-11eb-8264-0fda0c5d13a6)

이 이미지들은 음식_음료_커피 카테고리로 분류되어 있으며 이 때 label은 음식, 음료, 커피 입니다 (순서대로).

# Dataset Detail
- Dataset ID/개수: 107 classes, About 7,300 training images 2,300 testing images.

- Label 구성: 코드 참조: https://open.oss.navercorp.com/airush2021/1-3/issues/1


# Code spec
- 이미지 한장을 입력으로 받아 multi-class label (with order) 출력 - 자세한 내용은 코드 참조. (max 3개)
- 예를 들어 위의 커피 이미지이면 76 (음식) 22 (음료) 23 (커피) 이런 식으로 제출. (레이블 숫자도 예시입니다)

# Measuring

본 문제에서는 실제 application에서 활용을 하기 위한 특화된 measuring을 제공합니다. 각 이미지는 0~1 사이의 실수 값으로 measuring 됩니다.
예를 들어 A-B-C 클래스로 labeling 된 이미지가 있다면

(1) 전체 클래스를 모두 맞춘 경우에는 최고 점수입니다 (A,B,C). 각 이미지당 최고 1 점으로 판정됩니다.

(2) 상위 클래스를 맞춘 경우에는 부분점수가 있습니다. 부분점수는 맞춘 클래스 개수 / gt class의 개수 가 됩니다.

- (A-B-C) -> 이 경우 1 점입니다.
- (A-B) -> 이 경우 2/3 점 입니다.다
- (A) -> 이 경우에는 1/3 점이 됩니다.

Note: 점수는 중첨되지는 않습니다. 순서에 맞게 올려주세요 (Tip: training set을 이용한 Hierarchy 분석이 필요합니다).

(3) 전체 결과는 총 test image 점수의 평균으로 판정됩니다.


# Requirements and warning

(1) 본 문제에서는 ImageNet pre-trained 모델을 추가로 활용하는 것이 가능합니다. 공평성을 위해 본 챌린지에서는 pytorch 공식 pre-trained model과 [timm](https://rwightman.github.io/pytorch-image-models/)에서 지원하는 pre-trained 모델만을 허용합니다. 물론 scratch로 진행하셔도 상관은 없습니다. 그 외의 임의의 pre-trained 모델을 직접 올리는 형태는 제한됩니다. 제출하시는 코드는 운영진에서 검수 가능하니 유의하시기 바랍니다.

(2) model 은 32bit FP 모델을 사용해 주시기 바랍니다. 이미지 사이즈는 제한이 없습니다.

(3) 이미지 크기의 제한은 걸려있지 않지만 model의 FLops 제한이 걸려있습니다. 32bit FP 모델 기준 2GMac를 초과하지 않도록 해주세요. 이 이상 크기를 가지는 모델은 Reject될 예정입니다.

(4) Flops 계산은 https://github.com/sovrasov/flops-counter.pytorch 를 기준으로 진행될 예정입니다.

(5) 제공하는 이미지는 저작권이 별도로 있고, 추가 저장/활용/재배포가 금지되어 있습니다. 무단으로 이용 시 법적으로 처벌 대상이 될 수 있으니 의도치 않게 저장되거나 배포되지 않도록 반드시 주의를 부탁드립니다.


# Running Training code on NSML
```
nsml run -v -d airush2021-1-3 -g 1 -e main.py  -a "--lr 0.001 --num_epoch 3"
```

# Example of listing checkpoints of the session
```
nsml model ls YOURID/airush2021-1-3/1
```

# Running example of the submission on NSML
```
nsml submit -v YOURID/airush2021-1-3/1 10
```

# ETC

더 자세한 내용은 example code를 참조해 주세요.
