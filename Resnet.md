# Resnet Review

2015년에 ILSVRC 이미지 인식 대회에서 1위 한 논문으로서 에러율이 인간보다 낮아지기 시작했고

깊은 NetWork를 사용했다는 점에서 이때부터가 진짜 딥러닝이라고 하는 논문이기도 함

## Introducktion

이 논문에서는 딥뉴럴 네트워크는 학습시키기 어렵기 때문에 비교적 쉬운 잔여 학습 방법을 제안함

직전에 나온 VGG보다는 깊지만 덜 복잡하다고 말함

지금까지의 AlexNET등의 논문만으로 보았을 때 신경망이 깊어질수록 더 정확한 예측을 할거라고

생각되었기 때문에 Layer만 깊게 쌓으면 될것이라고 생각할 수 있지만 Layer가 깊어질 수록

Gradient Vanishing(기울기 소실)과 Gradient Exploding(기울기 폭발) 문제가 발생함

( Plain network는 skip/shortcut connection을 사용하지 않은 일반적인 CNN(AlexNet, VGGNet) 신경망을 의미함 )

신경망이 깊을 때, 작은 미분값이 여러번 곱해지면 0에 수렴하는데 이를 Gradient vanishing이라고 함

반대로, 큰 미분값이 여러번 곱해지면 값이 매우 커지는데 이를 Gradient Exploding이라고 함

아래 그림은 20-layer plain network가 50-layer plain network보다 더 낮은 train error와 test error를 얻은 것을 보여줌

논문에서는 이를 degradation 문제라고 말하고 Gradient vanishing에 의해 발생한다고 함

하단 사진은 이전의 연구들로 모델의 Layer가 깊어질수록 오히려 성능이 떨어짐을 증명한 사진임

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcyb9pL%2FbtqYur1rFVH%2FatPKJaR6i5xGgz9V6pek21%2Fimg.png)

---

## Skip / Shortcut Connection in Residual Network(ResNet)


![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbmdg7R%2FbtqYDjgD1TR%2Fp6qeoRgyJlJvBjKnTPNB9k%2Fimg.png)

상단의 Figure2는 ResNet 모델의 구조를 시각적으로 보여주는 그림임. 해당 그림은 ResNet-34라는 모델을 예시로 사용하고 있음

ResNet-34는 총 34개의 레이어로 이루어져 있고 이 중에서 첫 번째 레이어는 입력 이미지를 처리하는 convolutional layer임

이후에는 여러 개의 residual block이 반복되어 쌓여 있는데 각 residual block은 shortcut connection과 두 개의 convolutional layer로 구성되어 있습니다.

shortcut connection은 입력값을 바로 출력값에 더해주는 것으로 residual function F(x)를 학습하는 데 도움을 줌

각 residual block 내에서는 shortcut connection과 convolutional layer가 번갈아가면서 연결되어 입력값 x가 shortcut connection을 통해 출력값에 직접적으로 전달됨

마지막으로 ResNet-34 모델은 global average pooling layer와 fully-connected layer로 구성된 classifier 부분으로 마무리됨

이 부분은 네트워크가 이미지 분류 문제를 해결할 수 있도록 최종 출력값을 계산하는 역할을 함

Figure2 에서는 ResNet-34 모델의 구조를 간단하게 보여주며 이러한 구조가 깊은 네트워크에서도 gradient vanishing, gradient exploding 문제 없이 잘 동작함을 보여줌

---

## Gradient Vanishing과 Gradient Exploding

Gradient Vanishing과 Gradient Exploding은 딥러닝에서 깊은 네트워크를 학습시킬 때 발생하는 문제로서

이 문제를 해결하기 위해 논문에서는 두 가지 방법을 제안함

첫 번째는 Normalized Initialization이라는 방법임

이 방법은 각 레이어의 가중치(weight)를 초기화할 때 특정한 분포(예: 평균 0, 분산 1인 정규분포)로부터 샘플링한 후 이를 Normalize(정규화)하는 것임

이렇게 함으로써 각 레이어의 출력값이 일정한 분산을 가지게 되어 gradient vanishing과 exploding을 방지할 수 있다고 함

두 번째는 Intermediate Nrmalization Layers라는 방법임

이 방법은 네트워크 내에 Normalization Layer를 추가하여 각 레이어의 출력값을 Normalize하는 것임

이렇게 함으로써 Gradient가 Backpropagation 과정에서 일정한 크기를 유지하면서 전파될 수 있게 되어 Gradient Vanishing과 Exploding 문제를 해결할 수 있다고 함

논문에서는 이러한 두 가지 방법을 조합하여 ResNet 모델을 구성하였고 실험 결과 깊은 네트워크에서도 잘 동작함을 보여줌

## ResNet Architecture

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQfaUX%2FbtqYAtD1KcX%2FZdc4DLFzR9SoJYBlO6M1uK%2Fimg.png)

위 그림은 ResNet의 구조임

상단 구조는 34-Layer ResNet이며 plain network에 skip/short connection이 추가되어있음

중간 구조는 깊어진 34-Layer plain network, 맨 아래 구조는 VGG-19를 나타냄

- 상단의 projection shortcut connection은 입력값 x를 1x1 convolutional layer를 통해 차원을 맞춘 후 출력값에 더해주는 방식임
이 방식은 identity mapping보다 조금 더 많은 파라미터를 가지지만 성능 향상에 도움을 줄 수 있음

- 중간의 shortcut connection with identity mapping은 입력값 x를 그대로 출력값에 더해주는 방식으로
잔차 함수 F(x)를 학습하는 데 도움을 줌. 이 방식은 ResNet 모델에서 가장 기본적으로 사용되는 방법임

- 하단의 plain network는 shortcut connection을 사용하지 않은 일반적인 네트워크임
이 경우에는 깊은 네트워크에서 gradient vanishing과 exploding 문제가 발생할 가능성이 높음

위 그림은 이러한 세 가지 shortcut connection의 형태를 시각적으로 보여주고 있으며

이러한 방식들이 ResNet 모델의 성능 향상에 어떤 영향을 미치는지 실험 결과와 함께 보여줌

---

## Bottleneck Design

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FB5i5c%2FbtqYDjnmO9t%2F4mYzLdkp1eIeUUs68vkepK%2Fimg.png)

1x1 conv layers는 오른쪽 그림과 같이 신경망의 시작과 끝에 추가되며 이 기법은 NIN과 GoogLeNet에서 제안됨

1x1 conv는 신경망의 성능을 감소시키지 않고 파라미터 수를 감소시킴

bottleneck design으로 연산량을 감소시켜 34-layer는 50-layer ResNet이 되고

bottleneck design을 지닌 더 깊은 신경망이 있는데 ResNet-101과 ResNet-152 임

전체적인 구조는 아래에 나타나 있음

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbbk33p%2FbtqYxpoqUIf%2Fc9iP9l9LTmwv6VCfcXso9k%2Fimg.png)

위 이미지를 보면 VGG-16 보다 ResNet-152가 더 적은 연산량을 갖고 있는걸 알 수 있음

---

## Plain Nerwork VS ResNet

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqSLDE%2FbtqYE8y96aq%2FptTau1wCNqnedWlHZ4LL61%2Fimg.png)

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrUPNa%2FbtqYDj17YPx%2FLfgFTWCpN0qLPHw9u0P880%2Fimg.png)

plain network에서 Gradient vanishing 문제 때문에 18-layer의 성능이 34-layer보다 뛰어남을 알 수 있음

ResNet에서는 Gradient vanishing 문제가 skip connection에 의해 해결되어 34-layer의 성능이 18-layer보다 뛰어난걸 알 수 있음

깊지 않은 신경망에서는 Gradient vanishing 문제가 나타나지 않기 때문에 18-layer plain network와 18-layer ResNet network에는 차이점이 없음
