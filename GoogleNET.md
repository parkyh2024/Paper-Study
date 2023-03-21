# GoogleNET

## Introduction

해당 논문은 AlexNET의 1/12배의 파라미터를 사용하였음에도 더 정확했다.

Inception이라고 불리는 구조를 사용하였으며 Computer Vision분야의 효율적인 deep neural network architecture에 focus를 맞추었다.

## Related Work

network-in-network방법은 1x1 합성곱 필터와 ReLU 함수를 사용하는 방법으로써 이 논문의 저자들은 이 방법을 사용했다.
 
여기서의 1x1 필터는 차원을 감소시키는 모듈로 주로 사용되었는데 컴퓨팅 병목 현상을 제거하기 위해 사용되었다.

이것은 깊이가 증가할수 있게 해줄 뿐만 아니라, 성능 저하 없이 이들의 network의 넓이도 늘릴수 있게 해준다.

이게 사용되지 않았다면 네트워크의 크기가 제한되었을 것이다.
 
## Motivation and High Level Considerations

Deep neural networks의 성능을 확실히 개선시키는 방법은 network의 사이즈를 증가시는 것인데 이 방법은 두가지의 큰 단점을 가지고 있다.

일반적으로 크기가 클수록 파라미터의 수가 많아지므로 확대된 네트워크가 과대적합될 가능성이 커지며 특히 training set에 라벨링된 data가 제한될 경우 더욱 그렇다.

이건 중요한 장애 요소가 될수 있는데 왜냐하면 training set에 일일히 라벨을 다는것은 매우 힘든 일이기 때문이다.

또 다른 단점은 컴퓨팅 리소스의 사용이 대폭 증가한다는 것이다.

예를 들어 deep vision network에서 만약 두개의 합성곱 층이 연결되어 있다면 ***4배***의 연상량이 증가하게 된다.

이 두가지 문제를 해결하는 근본적인 방법은 완전히 연결된 아키텍쳐에서 sparsely connected architectures로 전환하는 것이다. 합성곱 층에서도 마찬가지이다.

## Architectural Details

