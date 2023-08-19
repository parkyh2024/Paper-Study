# R-FCN 리뷰

일반적으로 2-stage detector는 서로 다른 task를 수행하는 두 sub-network간에 주로 학습하는 속성에서 차이가 발생하는데 이를 translation invariance 딜레마라고 함

이러한 문제를 해결하기 위해 ResNet 논문의 저자는 모델 설계 시 conv layer 사이에 RoI pooling을 삽입했음

하지만 이같은 방법을 사용할 경우 수많은 RoI를 개별적으로 conv, fully connected layer에 입력시켜야 함

R-FCN 논문의 저자는 이로 인해 학습, 추론 시 많은 시간이 소요된다는 점을 지적

R-FCN 모델을 살펴보기에 앞서 논문을 이해하기 위해 필요한 배경 지식부터 가볍게 짚고 넘어가도록 하겠음

![이미지](https://user-images.githubusercontent.com/122156509/261748164-3fe8e018-a803-4c79-adeb-af29894578fb.jpg)
