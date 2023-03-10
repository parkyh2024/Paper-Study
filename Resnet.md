# Resnet Review

## Plain Network의 문제점
Plain network는 skip/shortcut connection을 사용하지 않은 일반적인 CNN(AlexNet, VGGNet) 신경망을 의미합니다. plaing network가 점점 깊어질 수록 기울기(gradient) 소실(vanishing)과 폭발(exploding) 문제가 발생합니다.

### 기울기 소실과 폭발
기울기를 구하기 위해 가중치에 해당하는 손실 함수의 미분을 오차역전파법으로 구합니다. 이 과정에서 활성화 함수의 편미분을 구하고 그 값을 곱해줍니다. 이는 layer가 뒷단으로 갈수록 활성화함수의 미분값이 점점 작아지거나 커지는 효과를 갖습니다. 신경망이 깊을 때, 작은 미분값이 여러번 곱해지면 0에 가까워 질 것입니다. 이를 기울기 소실이라고 합니다. 반대로, 큰 미분값이 여러번 곱해지면 값이 매우 커질 것입니다. 이를 기울기 폭발이라고 합니다. 신경망이 깊어질 수록 더 정확한 예측을 할 것이라고 생각할 수 있습니다. 하지만 아래 그림은 20-layer plain network가 50-layer plain network보다 더 낮은 train error와 test error를 얻은 것을 보여줍니다. 논문에서는 이를 degradation 문제라고 말하고 기울기 소실에의해 발생한다고 합니다. 







