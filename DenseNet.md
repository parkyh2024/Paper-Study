# DenseNet

![이미지](https://user-images.githubusercontent.com/122156509/231355963-0f8ecf16-226a-4509-8500-9aec3d54d26e.jpg)

 일반적인 CNN은 Layer가 깊어질수록 정확도는 향상되지만 연산량과 파라미터 수가 많아지므로
 
BackPropagation 시 Gradient Vanishing 문제가 발생함

![이미지](https://user-images.githubusercontent.com/122156509/231355970-20f2de17-fe62-4ffd-b9fd-55ecd83e00dc.jpg)

이걸 해결해기 위해 ResNet은 Residual Connection을 사용함 (= Skip-Connection)

---

![이미지](https://user-images.githubusercontent.com/122156509/231355972-751506d5-b02c-40f1-8a99-a684032cabdb.jpg)

DenseNet은 모든 Layer를 연결하고 DenseBlock을 여러개를 둔 형태임

---

![이미지](https://user-images.githubusercontent.com/122156509/231355983-88f0aebf-7391-4c86-b7f8-6be760427852.jpg)

세 개를 비교해 보면

CNN은 X의 출력으로 F(x)가 나오는데 F(x)가 Conv. ReLU가 반복되는 구조임

ResNet은 입력 X에 대한 출력인 G(x)와 입력 X를 합하는 구조로 G(x)가 BN + LeRU + Conv. 의 반복 구조임

DenseNet은 X의 출력인 H(x)와 입력 X를 Concatenate 하는 구조로 H(x)가 BN, ReLU, Conv., DropOust이 반복되는 구조임

---

### DenseNet

![이미지](https://user-images.githubusercontent.com/122156509/231355993-76b1a0c8-b1f5-4729-97b0-f2f1bbce7320.jpg)

이전 Layer의 입력을 다음 Layer의 출력에 Concatenate하는 구조로 K channel 수가 표를 아주 작게 유지하여

컴퓨터 연산량을 적게 하고 파라미터 수도 적어 슬림한 구조임

### Forward Propagation

![이미지](https://user-images.githubusercontent.com/122156509/231356003-92cd16ba-5190-4fbf-8a1a-127f03549d68.jpg)

X0를 H1이라는 Convolution을 거치면 나오는 출력이 X1 인데 여기에 X0를 Concatenate하고

X0와 X1이 H2라는 Convolution을 거쳐 나온 X2에 X0, X1을 Concatenate한다.

이런 Concatenate를 이후 Layer에도 반복하며

H4 단계를 보면 Composit Layer라고

![이미지](https://user-images.githubusercontent.com/122156509/231356011-42e1a60a-fc29-490b-be93-711fef0507c5.jpg)

이렇게 생겼는데 Batch Norm, ReLU, 3x3 Convolution으로 이루어져 있다

![이미지](https://user-images.githubusercontent.com/122156509/231356019-ddd52139-1e7b-4ca3-857c-c476ba8b61e6.jpg)

이런 Composit Layer를 반복하면 K Layer 수가 커지므로 Composit Layer인 Batch Norm, ReLU, 3x3 Convolution 이전에

BottlesNeck Layer인 Batch Norm, ReLU, 1x1 Convolution을 두어 Channel 수를 4xk 사이즈로 줄임으로서

파라미터 수를 감소시키고 컴퓨터 연산량 또한 감소시켰다.

![이미지](https://user-images.githubusercontent.com/122156509/231356030-4b0dff0e-03a0-4a3a-b406-fe7e1c681982.jpg)

DenseNet은 DenseBlock사이에 Transition Layer인 Convolution Pooling을 두어 이 Pooling이 Feature map Size를 줄이고

DenseBlock내에서는 Feature map Size를 동일하게 맞춤으로서 K Channel이 Concatenate 될 수 있도록 함

Dense Black 이 핵심임

---

## Advantage Of DenseNet

![이미지](https://user-images.githubusercontent.com/122156509/231356037-28e1e0a7-5de8-46ac-85c5-99bc2275f616.jpg)

Error Signal이 이전 layer로 잘 전파됨. 즉, Back Propagation이 잘됨

이유는 모든 layer가 연결되어 있기 때문

![이미지](https://user-images.githubusercontent.com/122156509/231356046-506d57d3-2662-4363-bf81-dc25b4cc3243.jpg)

컴퓨터 연산량이 감소함

ResNet을 보면 입력 C 채널과 H1 Convolution 결과인 C 채널을 합하므로 연산량이 CxC인데

DenseNet은 입력 Layer 수 x 채널수 K 의 H1 H1 Convolution 결과인 lxkxk가 연산량이다.

k 채널 수가 ResNet의 C 보다 매우 적으므로 DenseNet의 파라미터 수가 ResNet보다 훨씬 적다.

![이미지](https://user-images.githubusercontent.com/122156509/231356055-948315e8-6ae6-48a6-8987-8a4d28c9a73c.jpg)

마지막 장점은Classifier가 저수준의 feature를 사용한다는 것임

일반적인 CNN을 보면 이전 Layer단계의 저수준의 feature들이 Convolution을 거쳐서 마지막에 Classifier 에서 고수준의 feature들 만을

사용하게 되는데 아래 사진을 보면

![이미지](https://user-images.githubusercontent.com/122156509/231356061-dc5fc055-a32e-485c-a1a8-89d08fce21cd.jpg)

DenseNet은 모든 이미지들의 Layer가 연결되어 있으므로 저수준이라 고수준, 모든 Level의 Classifier를 사용하게 된다

따라서 분류 성능이 높다

![이미지](https://user-images.githubusercontent.com/122156509/231356080-e70eed1c-3c85-4892-b032-526de0ea1c82.jpg)

CIFAR-10 데이터를 사용하여 ResNet과 DenseNet의 분류 성능을 비교해보면 좌측 Data agumentation을 한 경우

ResNet, 1,001 Layers 10.2M 파라미터의 테스트 에러가 4.62%

DenseNet, 100 Layers 0.8M 파라미터의 테스트 에러가 4.5% 로

큰 차이가 없는 것을 확인 할 수 있다

즉, DenseNet은 모델의 크기가 작더라도 분류 성능이 높은걸 알수 있다.

Layer가 깊고 파라미터 수가 많은 DenseNet 250 Layers 15.3M 파라미터 수를 보면 좌측에 Test Error가 3.6%로

Previouis STA보다 성능이 좋은걸 알 수 있다.

Data agumentationdmf 한 경우 ResNet은 Test Error가 11.26%, 10.56% 로 OverFitting되었다.

반면에 DenseNet은 Previous STA보다도 분류 성능이 높게 나타난걸 확인할 수 있다.
