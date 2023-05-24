# MobileNet

2017년 4월 발표된 논문으로서 성능 저하를 최소화 하며 딥러닝 모델의 사이즈를 감소시키는 것을 주 목적으로 하여 발표됨

### 핵심 내용 3가지

- Depthwise separable convolution (Point)
- Width multiplier
- Resolution multiplier

---

![이미지](https://user-images.githubusercontent.com/122156509/240445450-d70e8f6b-ce99-4e3f-ae40-344c470e873a.png)

(a)는 보통 우리가 딥러닝에서 사용하는 conv 필터임

근데 여기에 사용되는 파라미터 수가 너무 많으니까 모바일넷에서는 그 갯수를 줄여보고자 함

(b) 처럼 가로세로 유지하고 채널만 1짜리를 적용해서 입력의 채널을 분리해서 낱개의 필터로 나눈 다음에 이 결과물에 1x1 conv를 적용해서

기존 방식의 출력물인 (a)와 모양이 같게 만듬. 이 과정에서 사용되는 파마미터 수가 줄어든다는 논문임

---

일반적으로 진행되는 CNN의 경우를 보면

![이미지](https://user-images.githubusercontent.com/122156509/240447340-3f0cc7a5-e722-4ccb-a3bb-2376398a1a3a.png)

위 입력에 일반적으로 사용하는 필터를 적용하면 아래와 같이 결과물이 나오는데

![이미지](https://user-images.githubusercontent.com/122156509/240448164-90acc341-4634-416f-a95d-21d31aa5d5b3.png)

이걸 합치면 아래처럼 됨

![이미지](https://user-images.githubusercontent.com/122156509/240447404-4ee5f2b3-3c6f-41f8-8363-2232ae437ccf.png)

그래서 여기에 1x1 conv를 진행 함

![이미지](https://user-images.githubusercontent.com/122156509/240447428-74dd586d-2b89-4e99-a679-5d31593d55ae.png)

여기까지의 연산량은 Dk x Dk x M x N x Df x Df 임. 이게 일반 CNN의 경우고

다음은 본 논문에서 말하는 Depthwise Separable Convolution을 봅시다.

![이미지](https://user-images.githubusercontent.com/122156509/240447340-3f0cc7a5-e722-4ccb-a3bb-2376398a1a3a.png)

입력은 아까와 동일함

![이미지](https://user-images.githubusercontent.com/122156509/240448481-756c462e-f187-4a7b-aceb-c8ca57286d15.png)

근데 기존 CNN과는 다르게 위와 같은 필터를 씀.

입력의 채널 하나하나마다 쓰는데 첫번째 필터는 입력의 첫번째 채널에, 두번째 필터는 입력의 두번째 채널에 대응되어 사용되는 방식임

![이미지](https://user-images.githubusercontent.com/122156509/240448719-9ddaaf3f-a0d3-48d6-b882-bb28fdeea02d.png)

그렇다면 위와 같은 중간 결과물이 나오는데 여기에 1x1 conv를 진행함

![이미지](https://user-images.githubusercontent.com/122156509/240448898-52e89fb9-b661-4885-bd38-bc88f888b0cc.png)

그러면 위와 같은 이미지 처럼 진행을 시키면

![이미지](https://user-images.githubusercontent.com/122156509/240449053-23a28c40-01c4-4c87-a282-5fecbc1a8037.png)

출력되는 결과물은 위와 같은데 

맨 위 사진의 (c)를 보면 이런 필터를 N개 진행한다고 되어있음

![이미지](https://user-images.githubusercontent.com/122156509/240449225-7db10b61-2845-4146-9545-45cac6367feb.png)

진행 하면 위와 같은 최종 결과물이 나옴

이 방식의 연산량을 보면 Dk x Dk x M x Df x Df + M x N x Df x Df

앞의 Dk : 중간 결과물 뽑을때 각 채널별로 적용하는 필터의 가로세로(채널은 1이니 생략)

다음 Df : 입력의 면적

M, N : 필터 1 x 1 x M 이 N 개 있으니까

Df : 중간결과의 가로세로

그래서 일반 CNN보다 연산량이 얼마나 줄었냐면 뒤에도 나오지만

1/N + 1/Dk^2 만큼 줄었다고 함

여기까지가 핵심 내용임

---

다음은 부가적인 내용인데

Width Multiplier : 입력 출력 채널을 a배 만큼 축소하여 진행(출력채널이 64이고 a=0.25라면 축소된 채널은 16)

Resolution Multiplier : 입력 및 중간 레이어들의 해상도를 b배 만큼 축소하여 진행(입력의 해상도가 224x224이고 b=0.571 이면 축소된 해상도는 128x128)

---

본문의 결과를 보면 

![이미지](https://user-images.githubusercontent.com/122156509/240449976-74706637-c167-41ac-b0aa-932db599c3d9.png)
![이미지](https://user-images.githubusercontent.com/122156509/240449993-51de111f-1ec9-4f9d-b4ad-7389a70ebaeb.png)



