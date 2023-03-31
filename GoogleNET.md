# GoogleNET

## Introduction

해당 논문은 2014 ILSVRC Competition 우승작으로서 AlexNET의 1/12배의 파라미터를 사용하였음에도 더 정확했다.

Inception이라고 불리는 구조를 사용하였으며 Computer Vision Part의 효율적인 Deep Neural Network Architecture에 중점을 두었다.

## Related Work

Network-in-Network방법은 1x1 합성곱 필터와 ReLU 함수를 사용하는 방법으로써 이 논문의 저자들은 이 방법을 사용했다.
 
여기서의 1x1 필터는 차원을 감소시키는 모듈로 주로 사용되었는데 컴퓨팅 병목 현상을 제거하기 위해 사용되었다.

이것은 깊이가 증가할수 있게 해줄 뿐만 아니라, 성능 저하 없이 이들의 Network의 넓이도 늘릴수 있게 해준다.

이게 사용되지 않았다면 네트워크의 크기가 제한되었을 것이다.
 
## Motivation and High Level Considerations

 Deep Neural Networks의 성능을 확실히 개선시키는 방법은 Network의 사이즈를 증가시는 것인데 이 방법은 두가지의 큰 단점을 가지고 있다.

 일반적으로 크기가 클수록 파라미터의 수가 많아지므로 확대된 네트워크가 Overfitting이 발생할 우려가 있다.

 이건 중요한 장애 요소가 될수 있는데 왜냐하면 training set에 일일히 라벨을 다는것은 매우 힘든 일이기 때문이다.

 또 다른 단점은 컴퓨팅 리소스의 사용이 대폭 증가한다는 것이다.

 예를 들어 deep vision network에서 만약 두개의 합성곱 층이 연결되어 있다면 ***4배의 연상량이 증가***하게 된다.

 이 두가지 문제를 해결하는 근본적인 방법은 완전히 연결된 아키텍쳐에서 sparsely connected architectures로 전환하는 것이다. 합성곱 층에서도 마찬가지이다.

 ### inception module을 통해 두 가지 효과를 얻을 수 있다.

1. dimension reduction을 통해 다음 계층의 input의 수를 조절할 수 있기 때문에 계산 복잡도에서 계산양의 증가없이 각 단계에서 unit의 수를 상당히 증가시킬 수 있다.
2. 1x1, 3x3, 5x5 convolution 연산을 통해 다양한 특징을 추출할 수 있기 때문에 시각 정보가 다양한 규모로 처리되고 다음 계층은 동시에 서로 다른 규모에서 특징을 추출할 수 있게 된다.

## GoogleNet

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdf2mjD%2Fbtrac795XSx%2F0fKaH0JLiGOWkhqbKbfHb1%2Fimg.png)

위 사진을 보면 아래가 Input, 위가 Output이다

아래쪽 레이어는 STEM레이어인데 convolusional레이어 라고도 불린다. 각 중간에 뭉쳐있어 보이는건 inception Block이고

중간에 있는 두가지는 Output인데 loss계산이 가능한 지점이다.

즉 중간 output지점에서 input쪽으로 backpropagation을 전달해 줄 수 있다는 의미이기도 하다.

물론 최종 output에서도 backpropagation을 전달하기도 한다.

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FA4nO0%2Fbtq98dKbKai%2FCmgKHK0GeUFjdmnfPPXuW0%2Fimg.png)


위의 표는 그래프(또는 다이어그램)를 테이블로 정리한 것이다. 많은 딥러닝 논문들이 이러한 표 형태로 자신들의 네트워크 구조를 표현한다.

우선 input이 들어가서 맨 위의 conv으로 들어갈테고 conv - maxpool - conv - maxpool 단계를 거치는데 이걸 STEM영역 또는 Conv영역이라고 한다.

첫 단계로 7x7 conv을 진행한다. 옆 숫자 2는 stide를 의미하며 stride 2를 줘서 진행한다는 의미다.

입력된 이미지 224x224x3 (컬러이미지니까 3레이어)이 첫 conv를 거쳐112x112x64가 된다.

depth는 진행하는 횟수를 의미하며 우측의 params는 파라미터인데 이 conv를 학습하기 위한 weight의 갯수를 의미한다. 

## Architectural Details

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbk1AHm%2Fbtrac9fKzpS%2FHrK8Qa9mptVaXgvt6HGuHK%2Fimg.png)

 Inception module을 보면 1x1 convolution이 보이는데 ***이것이 inception module의 핵심***이다.
 
 1x1 convolution의 목적은 dimmension reduction을 적용하여 필요한 연산량을 감소시키는 것이다.
 
 3x3와 5x5 convolution 연산 이전에 1x1 convolution이 적용되었는데 이는 dimmension reduction을 통해 input filter의 수를 조절하기 위함입니다.
 
 예를 들어, 이전 layer에서 256개의 channel을 가진 output이 생성되었다면 64개의 1x1 convolution filter를 이용해서 64 channel로 줄일 수 있습니다.
 
 이를 통해 다양한 크기의 filter(1x1, 3x3, 5x5)를 적용하여 여러 특징을 추출하지만 연산량을 낮출 수 있게 됩니다.
 
 하지만 모든 layer에서 inception module이 이용되는 것은 아니고 효율적인 메모리 사용을 위해 낮은 layer에서는 기본적인 convolution layer를 적용하고
 
 높은 layer에서는 inception module을 사용한다.

