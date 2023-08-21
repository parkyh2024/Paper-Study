# R-FCN 리뷰

일반적으로 2-stage detector는 서로 다른 task를 수행하는 두 sub-network간에 주로 학습하는 속성에서 차이가 발생하는데

이를 translation invariance 딜레마라고 함

이러한 문제를 해결하기 위해 ResNet 논문의 저자는 모델 설계 시 conv layer 사이에 RoI pooling을 삽입했음

하지만 이같은 방법을 사용할 경우 수많은 RoI를 개별적으로 conv, fully connected layer에 입력시켜야 함

R-FCN 논문의 저자는 이로 인해 학습, 추론 시 많은 시간이 소요된다는 점을 지적함

---

![이미지](https://user-images.githubusercontent.com/122156509/261748164-3fe8e018-a803-4c79-adeb-af29894578fb.jpg)

R-FCN 모델의 구성은 backbone network와 RPN(Region Proposal Network)임

backbone network는 feature extract 기능을 수행하며, 논문에서는 ResNet-101 모델을 사용함

원본 이미지를 backbone network와 RPN에 입력하여 각각 K^2(C+1)-d channel을 가지는 Position-sensitive score maps과 RoI(Region of Interest)를 얻음

이를 활용하여 Position-sensitive RoI pooling을 수행하여 kxk(x(C+1)) 크기의 feature map을 출력함

feature map의 각 channel별로 요소의 평균값을 구하는 voting을 수행하여 (C+1) 크기의 feature vector를 얻고 이에 대하여 softmax 함수를 적용하여 loss를 계산함

---

## Main Ideas

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/047b979d-1bfc-4dd8-bf13-f290b46113f1)

Translation invariance는 입력값의 위치가 변해도 출력값은 동일할 경우에 해당하는 함수의 속성인데

만약 위치가 서로 다른 동일한 객체, 예를 들어 위의 그림과 같이 석상(statue)의 이미지를 특정 모델에 입력해도 동일하게 석상이라고 인식할 경우

해당 모델은 translation invariance한 속성이다 라고 할 수 있고,

반대로 입력값의 위치가 변하면 출력값이 달라질 경우 이를 translation variance(=equivalence)라고 함

 

Image classification task 시, 이미지 내 객체의 위치가 바뀌더라도 동일한 객체로 인식하는 것이 바람직하기 때문에

인식 image classification 모델은 translation invariance 속성을 선호하고

반면 Object detection 시에는 객체의 위치가 변화하면 이러한 변화를 잘 포착하는 것이 바람직하기 때문에 학습 시 translation variance 속성을 중요시 함

 

2-stage detector의 경우 feature를 추출하는 역할을 수행하는 backbone network와 detection을 수행하는 network로 구성되어 있음

그중 backbone network는 image classification task를 위해 pre-trained되어 있음

R-CNN의 경우 AlexNet, Fast R-CNN과 Faster R-CNN의 경우 VGG16이 backbone network임

즉 원본 이미지를 backbone network에 입력하여 얻은 feature map은 translation invariance한 속성을 띄고 있음

 

반면 detection을 수행하는 network는 translation variance한 속성을 가져 객체의 위치 변화에 민감하게 반응하는 것이 바람직한데

원본 이미지를 backbone network에 입력하여 얻은 feature map은 위치 정보가 소실된 채로 detection network로 입력됨

detection network는 객체에 대한 위치 정보가 부재한 feature map이 입력되어 적절하게 학습이 이뤄지지 않음

이처럼 두 network간에 충돌이 발생하는 경우를 translation invariance dilmma라고 하며, 이로 인해 mAP값이 하락하게 됨

---

![이미지](https://user-images.githubusercontent.com/122156509/261754145-0cbb2dc5-66df-4e70-9e29-1b2cc755aff5.png)

ResNet 논문의 저자는 위와 같은 문제를 해결하기 위해 두 conv layer 사이에 RoI pooling layer를 추가했음(ResNet은 classification task 외에도 object detection task용으로도 활용될 수 있있음).

Object detection task를 위해 설계된 ResNet의 구조는 backbone network로 ResNet을 사용하며, 전체적인 구조는 Faster R-CNN 모델과 유사함

하지만 backbone network 이후 conv1~4라는 conv layer가 있으며, RoI pooling 이후 conv5라는 conv layer가 있다는 점에서 차이가 있음

ResNet+Faster R-CNN 모델은 두 conv layer 사이에 RoI pooling을 삽입하여 region specific한 연산을 추가함

이는 network가 서로 다른 위치에 있는 객체를 서로 다르게 인식한다는 것을 의미하고 이를 통해 RoI pooling layer 이후 conv layer는 translation variance한 속성을 학습하는 것이 가능해짐

 

하지만 본 논문의 저자는 ResNet+Faster R-CNN 모델과 같은 방법을 사용할 경우 성능은 높일 수 있지만 모든 RoI를 개별적으로 conv, fc layer에 입력하기 때문에 학습 및 추론 속도가 느려진다는 점을 지적함

이러한 문제를 해결하기 위해 R-FCN 모델은 RPN을 통해 추출한 RoI끼리 연산을 공유하면서 객체의 위치에 대한 정보를 포함한 feature map을 사용하는 구조를 가지고 있음

---

## Backbone Network

R-FCN 모델은 backbone network로 ResNet-101 network를 사용함

논문의 저자는 pre-trained된 ResNet-101 모델의 average pooling layer와 fc layer를 제거하고 오직 conv layer만으로 feature map을 연산하도록 학습시킴

마지막 feature map의 channel은 2048-d이며, 1x1 conv 연산을 적용하여 channel 수를 1024-d로 줄임

## Position sensitive score maps & Position-sensitive RoI pooling

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/00cd260f-8750-4c86-b0e6-c079e08ca5c4)

RPN을 통해 얻은 각각의 RoI에 대하여 class별로 위치 정보를 encode하기 위하여 RoI를 k x k 구간의 grid로 나눠줍니다. RoI의 크기가 w x h 인 경우, 각 구간의 크기는 대략적으로
w/k x h/k 임

논문에서는 k = 3 으로 지정함

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/d0b3f8c5-109c-4b93-a584-c427a275462a)

앞서 얻은 feature map의 channel 수가 k^2(C+1)이 되도록 마지막 conv연산을 적용하여 Position-sensitive score map을 생성함

여기서 C = class의 수 를 의미함(배경을 포함하기 때문에 1을 더해줌)

이같은 경우 RoI를 9(K^2=9)개의 구간으로 나눠 class별로 위치 정보인 {top-left, top-center, top-right, ..., bottom-right}에 해당하는 정보를 encode하고 있다고 볼 수 있음

Position-sensitive score map과 RoI를 활용하여 (i, j) 번째 구간에서 오직 (i, j)번째 score map만 pooling하는 Position-sensitive RoI pooling을 수행함

pooling한 결과를 구하는 수식은 아래와 같음

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/cd10d6ab-39f6-4317-9c27-c39b1963555d)

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/99632fe7-91de-4cd8-b562-f3873d75fecf)

간단하게 살펴보면 각 class별로 w/k x h/k 만큼의 RoI grid에 대하여 average pooling을 수행한 것이라고 볼 수 있습니다. 이를 통해 RoI별로 크기가 k x k 이며

channel 수가 (C+1)인 feature map이 생성됨

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/9700cddc-d1df-46e3-ad9f-7cd58329db99)

이후 각 class별로 k x k 크기의 feature map의 각 요소들의 평균을 구함

논문에서는 이 과정을 voting이라고 언급하며 k = 3 일 경우, channel별로 9개의 요소의 합의 평균을 구하면 됨

이를 통해 (C+1)크기의 feature vector를 얻을 수 있고, softmax function을 통해 loss를 계산함

위의 그림은 position-sensitive RoI pooling과 voting을 수행하는 과정의 그림임

---

논문에서는 bounding box regression 역시 비슷한 방법으로 수행함

K^2(C+1)-d feature map 외에도 4k^2-d feature map을 추가하여 bounding box regression을 수행함

이에 대한 내용은 아래 Training 파트에서 살펴보겠음

## Loss function

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/70867748-9fa0-429e-a072-1b939eab52d4)

Loss function은 Fast R-CNN 모델과 같이 cross-entropy loss와 bounding box regression loss의 합으로 구성되어 있음

여기서 c∗은 RoI의 ground truth label에 해당하며, IoU 값을 기준으로 0.5 이상일 경우 c∗=1, 그 이외의 경우에는 c∗=0 임

두 loss 사이의 가중치를 조절하는 balancing parameter인 λ=1로 설정함

## Training

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/3577201c-5a2a-43c7-9cc5-6d56700a290f)

### 1) feature extraction by pre-trained ResNet-101

원본 이미지를 pre-trained된 ResNet-101 모델에 입력하여 feature map을 얻음

 * Input : image
 * Process : feature extraction
 * Output : feature map

### 2) Position-sensitive score maps by conv layer

앞서 얻은 feature map을 channel 수가 k^2(C+1)이 되도록 하는 conv layer에 입력하여 Position-sensitive score maps를 얻음

논문에서 k=3, C=20으로 지정했고 bounding box regression 역시 이와 같은 방법으로 수행함

다만 앞서 얻은 feature map을 channel이 4k^2가 되도록 하는 conv layer에 입력함

이를 통해 RoI의 각 구간별로 bounding box의 offset이 encode된 4k^2-d feature map을 얻음

 * Input : feature map
 * Process : 3x3(xk2(C+1)) conv layer, 3x3(x4k2) conv layer
 * Output : k^2(C+1)-d feature map(position-sensitive score map), 4k^2-d feature map

### 3) Region proposal by RPN

원본 이미지를 pre-trained된 ResNet-101 모델에 입력하여 얻은 feature map을 RPN(Region Proposal Network)에 입력함

이를 통해 RoIs를 얻을 수 있음

 * Input : feature map from pre-trained ResNet-101
 * Process : region proposal
 * Output : RoIs

### 4) Average pooling by Position-sensitive pooling

2)번 과정에서 얻은 k^2(C+1)-d feature map(position-sensitive score map),4k^2-d feature map과

3)번 과정에서 얻은 RoIs를 사용하여 Position-sensitive pooling을 수행함

이 과정을 통해 각각 k×k(×(C+1)) feature map과 k×k(×4) 크기의 feature map을 얻을 수 있음

 * Input : k^2(C+1)-d feature map(position-sensitive score map),4k^2-d feature map and RoIs

 * Process : position-sensitive pooling

 * Output : k×k(×(C+1)) sized feature map, k×k(×4) sized feature map
 
### 5) Voting

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/0a895723-f4c4-45bf-9f78-9d784fa4f586)

4)번 과정을 통해 얻은 feature map에 대하여 각 channel의 요소들의 평균을 구하는 voting 과정을 수행하고

이를 통해 k×k(×(C+1)) 크기의 feature map으로부터 class score에 해당하는 (C+1) 크기의 feature vector를,

k×k(×4) 크기의 feature map으로부터 bounding box regressor에 해당하는 길이가 4인 feature vector를 얻을 수 있음

 * Input : k×k(×(C+1)) sized feature map, k×k(×4) sized feature map
 * Process : Voting
 * Output : (C+1)-d sized feature vector, 4-d sized feature vector 

### 6) Train R-FCN network by loss function

마지막으로 앞선 과정에서 얻은 feature vector를 사용하여 각각 cross-entropy, smooth l1 loss를 구한 후 backward pass를 통해 network를 학습시킴

실제 학습 시에는 RPN과 R-FCN을 번갈아가며 학습하는 4-step alternating training 방식을 사용했다고 함

---

# 결론

R-FCN 모델은 class별로 객체의 위치 정보를 encode한 position-sensitive score & pooling을 통해 translation invariance dilemma를 효과적으로 해결했음

이를 통해 PASCAL VOC 2007 데이터셋을 사용했을 때, 83.6%라는 높은 mAP값을 보여줌

R-FCN 모델은 이름 그대로 fully convolutional network이며, 오직 conv layer로만 구성됨

또한 position-sensitive pooling 이후 학습 가능한 layer가 없기 때문에 region-wise 연산량이 많지 않아(cost free) 학습 및 추론 속도가 빠름

detection 시 이미지 한 장당 170ms 정도 소요되며 이는 ResNet + Faster R-CNN 모델보다 0.5~20배 이상 빠른 속도라고 함함
