# Resnet Review

## Plain Network의 문제점

Plain network는 skip/shortcut connection을 사용하지 않은 일반적인 CNN(AlexNet, VGGNet) 신경망을 의미함

근데 Plain network가 점점 깊어질 수록 Gradient vanishing(기울기 소실)과 Gradient Exploding(기울기 폭발) 문제가 발생함

---

### Gradient vanishing 과 Exploding
Gradient를 구하기 위해 가중치에 해당하는 손실 함수의 미분을 Backpropagation으로 구함

이 과정에서 활성화 함수의 편미분을 구하고 그 값을 곱해줌

이는 layer가 뒷단으로 갈수록 활성화함수의 미분값이 점점 작아지거나 커지는 효과를 냄

신경망이 깊을 때, 작은 미분값이 여러번 곱해지면 0에 수렴하는데 이를 Gradient vanishing이라고 함

반대로, 큰 미분값이 여러번 곱해지면 값이 매우 커지는데 이를 Gradient Exploding이라고 함

신경망이 깊어질 수록 더 정확한 예측을 할 거라고 생각할 수 있는데

아래 그림은 20-layer plain network가 50-layer plain network보다 더 낮은 train error와 test error를 얻은 것을 보여줌

논문에서는 이를 degradation 문제라고 말하고 Gradient vanishing에 의해 발생한다고 함

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcyb9pL%2FbtqYur1rFVH%2FatPKJaR6i5xGgz9V6pek21%2Fimg.png)

---

## Skip / Shortcut Connection in Residual Network(ResNet)

Gradient vanishing / Gradient Exploding 문제를 해결하기 위해 입력 x를 몇 layer 이후의 출력값에 더해주는 skip/shortcut connection을 더해줌

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbmdg7R%2FbtqYDjgD1TR%2Fp6qeoRgyJlJvBjKnTPNB9k%2Fimg.png)

기존의 신경망은 H(x) = x가 되도록 학습 했고 skip connection에 의해 출력값에 x를 더하고 H(x) = F(x) + x로 정의하고

그리고 F(x) = 0이 되도록 학습하여 H(x) = 0 + x가 되게 함. 이 방법이 최적화하기 훨씬 쉽다고 함

미분을 했을 때 더해진 x가 1이 되어 Gradient vanishing 문제가 해결됨

이 문제가 해결되면 정확도가 감소되지 않고 신경망의 layer를 깊게 쌓을 수 있어 더 나은 성능의 신경망 구축 가능

---

## ResNet Architecture

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbQfaUX%2FbtqYAtD1KcX%2FZdc4DLFzR9SoJYBlO6M1uK%2Fimg.png)

위 그림은 ResNet 구조인데 맨 아래 구조는 VGG-19 를 나타냄

중간 구조는 VGG-19가 더 깊어진 34-layer plain network 를 나타냄

맨 윗 구조는 34-layer residual network(ResNet)이며 plain network에 skip/short connection이 추가되었음

skip/short connection을 추가하기 위해서는 더해지는 값x와 출력값의 차원이 같아야 하며

ResNet에서는 입력 차원이 출력 차원보다 작을 때 사용하는 3종류의 skip/shortcut connection이 있음

A. Shortcut은 증가하는 차원에 대해 추가적으로 zero padding을 적용하여 identity mapping을 수행하므로 추가적인 파라미터는 없음

B. 차원이 증가할 때만 projection shortcut을 사용하고 다른 shortcut은 identity 이므로 추가적인 파라미터가 필요함

C. 모든 shortcut이 projection 임. B 보다 많은 파라미터가 필요함

모델의 연산량이 증가하기 때문에 이 논문에서 C는 사용하지 않음

아래에 있는 bottleneck 구조에서 A 옵션을 사용함

---

## Bottleneck Design

신경망이 깊어지면 학습하는데 소요되는 시간은 크게 증가함

bottleneck design은 다음과 같이 신경망의 복잡도를 감소하기 위해 사용됨 

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
