# SqueezeNet 리뷰

## 개요 

2016년 이미지 대회에서 우승한 논문으로 딥러닝 모델의 파일 크기는 줄이고 성능은 유지하는게 목적이다

- 파일크기를 줄이면 갖는 이점

  * 분산환경에서의 학습 용이
  * 말단 기기로 모델파일 전송 유리 (자율주행차 등)
  * 저사양 회로에서 사용 가능 (메모리 10MB이하인 FPGA등)

### 해당 논문에서 모델 크기를 줄인 방법

![이미지](https://velog.velcdn.com/images/twinjuy/post/372c6dc6-0509-4979-9206-c7d5a08ac79e/image.png)

상단 사진은 Squeeze layer, Expand layer로 구성된 Fire module에 대한 개략도이다

---

### 본론

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtFgTS%2Fbtq0bv0uqr4%2FlR9KbOa8C62j1qVPD99t4K%2Fimg.png)

첫번째 이미지(224x224x3)가 입력이 되고 7x7 conv 필터를 거치는데, 보통 3x3 또는 5x5를 쓰는데 이건 좀 큰 필터를 사용한다

거치고 나면 111x111x96이 출력된다

다음은 맥스 풀링(3x3)이 진행된다. 보통 2x2를 쓰지만 이것도 큰 필터를 사용하는 편이다 이후 55x55x96이 출력되고

**두번 째 구간에 들어선다.** 이게 Fire Module 구간이며

진입 후 1x1 conv 필터를 거친다. 보통 3x3 이나 5x5를 쓰는데 1x1 conv 필터를 쓴다

3x3필터를 사용하면 각각의 파라미터가 9개가 되는데 1x1을 사용하면 파라미터 수가 1개로 줄어들어 1/9 배로 감소한다

다음으로 1x1 conv 필터를 적용하는데 입력이 55x55였기에 출력도 55x55가 된다

fire module을 통과하고 나면 채널 수가 16개가 되고 55x55x16이 된다

다음은 Expand layer에 진입하는데 표를 보면 1x1 도 있고 3x3도 있다

1x1의 경우는 가로세로는 같지만 64개가 있으니 55x55x64가 출력되고

3x3의 경우는 제로패딩이 되고 갯수는 동일하게 64개가 있으니 55x55x64가 출력된다

두 출력물의 가로세로가 같으므로 합쳐주면 55x55x128이 되며 여기까지가 fire module이다

**이제 세번 째 구간에 들어서며** 진입할 때의 결과물은 13x13x512가 입력된다

진입 후 1x1 conv을 진행하는데 (1000개인 이유는 이미지넷이 이미지 1000개를 주기 때문)

1x1 conv을 거치면 13x13x1000이 나오고 13x13에 대해 EvgPooling을 한다

그러면 슬라이스 하나하나 값이 한 개가 나오고 이게 1000개가 있게 됨 (1x1x1000)

그래서 각 항목마다의 점수가 들어있고 가장 점수가 높은것으로 예측을 진행함

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbyXeKq%2Fbtq0dn1VfUy%2FEb47nMXmJIMOf4beyZvWKK%2Fimg.png)

여태 한게 좌측 그림을 설명한거고 이외에 성능 향상을 위한 기법이 두 개 더 있음

가운데 방법처럼 중간중간 정보를 다음칸으로 전달 해주면 좌측 방식보다 2~3%정도 정확도가 높아졌다고 함

---

## 성능 비교 도표

![이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FIa3hM%2Fbtq0bxxgXcQ%2FjhnfqsQl8h2KnBGsFqHkK0%2Fimg.png)

알렉스넷과 비교를 했는데 알렉스넷은 240MB 였고 스퀴즈넷은 4.8MB 임. 그럼에도 용량은 1/50 배까지 감소함

추가적으로 압축 알고리즘 적용시 0.66MB, 0.47MB까지 용량을 줄였다고 함

또한 알렉스넷의 정확도는 57.2%였으며 스퀴즈넷은 파일크기를 1/50배로 감소 시켰음에도 57.5%라는 비슷하거나 좀더 높은 정확도를 
