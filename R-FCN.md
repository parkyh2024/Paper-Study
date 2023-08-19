# R-FCN 리뷰

일반적으로 2-stage detector는 서로 다른 task를 수행하는 두 sub-network간에 주로 학습하는 속성에서 차이가 발생하는데

이를 translation invariance 딜레마라고 함

이러한 문제를 해결하기 위해 ResNet 논문의 저자는 모델 설계 시 conv layer 사이에 RoI pooling을 삽입했음

하지만 이같은 방법을 사용할 경우 수많은 RoI를 개별적으로 conv, fully connected layer에 입력시켜야 함

R-FCN 논문의 저자는 이로 인해 학습, 추론 시 많은 시간이 소요된다는 점을 지적

R-FCN 모델을 살펴보기에 앞서 논문을 이해하기 위해 필요한 배경 지식부터 가볍게 짚고 넘어가도록 하겠음

---

![이미지](https://user-images.githubusercontent.com/122156509/261748164-3fe8e018-a803-4c79-adeb-af29894578fb.jpg)

R-FCN 모델의 구성은 backbone network와 RPN(Region Proposal Network)임

backbone network는 feature extract 기능을 수행하며, 논문에서는 ResNet-101 모델을 사용함

원본 이미지를 backbone network와 RPN에 입력하여 각각 K^2(C+1)-d channel을 가지는 Position-sensitive score maps과 RoI(Region of Interest)를 얻음

이를 활용하여 Position-sensitive RoI pooling을 수행하여 kxk(x(C+1)) 크기의 feature map을 출력함

feature map의 각 channel별로 요소의 평균값을 구하는 voting을 수행하여 (C+1) 크기의 feature vector를 얻고 이에 대하여 softmax 함수를 적용하여 loss를 계산함

---

### Main Ideas

이제 본 논문의 핵심내용을 보도록 하겠음

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/047b979d-1bfc-4dd8-bf13-f290b46113f1)

Translation invariance는 입력값의 위치가 변해도 출력값은 동일할 경우에 해당하는 함수의 속성인데

만약 위치가 서로 다른 동일한 객체, 예를 들어 위의 그림과 같이 석상(statue)의 이미지를 특정 모델에 입력해도 동일하게 석상이라고 인식할 경우

해당 모델은 translation invariance한 속성을 가지고 있다고 할 수 있고

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

### Backbone Network

R-FCN 모델은 backbone network로 ResNet-101 network를 사용함

논문의 저자는 pre-trained된 ResNet-101 모델의 average pooling layer와 fc layer를 제거하고 오직 conv layer만으로 feature map을 연산하도록 학습시킴

마지막 feature map의 channel은 2048-d이며, 1x1 conv 연산을 적용하여 channel 수를 1024-d로 줄임

### Position sensitive score maps & Position-sensitive RoI pooling

![이미지](https://github.com/parkyh2024/Paper-Study/assets/122156509/00cd260f-8750-4c86-b0e6-c079e08ca5c4)

RPN을 통해 얻은 각각의 RoI에 대하여 class별로 위치 정보를 encode하기 위하여 RoI를 k x k 구간의 grid로 나눠줍니다. RoI의 크기가 w x h 인 경우, 각 구간의 크기는 대략적으로
w/k x h/k 임

논문에서는 k = 3 으로 지정함

