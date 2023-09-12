## YOLO Version 2

> 핵심 특징
  * Grid Cell 당 Anchor Box 수 : 5
  * Anchor box 사용 - 결정 방법 : K-means Clustering. Anchor box 크기와 ratio를 정하는 방법. GT Box를 그룹핑 한 후, 그것을 이용해서 Anchor box 크기와 ratio 결정.
  * Output Peature Map 크기 : 13 x 13
  * Batch Normaliztion 을 적극적으로 활용하기 시작
  * High Resolution Classifier : 처음에 Input 224 x 224로 학습하다가, input 448 x 448로 Fine tuning 하여 성능향상을 얻음.
  * 서로 다른 크기의 Image들 섞어서 배치하여 작은 Object도 Detect할 수 있도록 노력.

---

> Network 구조도

![이미지](https://user-images.githubusercontent.com/122156509/267206109-96f29df4-9ef5-4505-b832-815132c8bcac.png)
  * Version 1에서 사용되던 Fully connected 구조가 사라짐

---

> Anchor Box

![이미지](https://user-images.githubusercontent.com/122156509/267207272-0ed57b5a-8a82-4b6d-b8a6-c90ee0b8eba4.png)
  * 1개의 Cell에서 여러개의 Anchor를 통해 개별 Cell에서 여러개 Object Detection이 가능해졌다. 이전에는 Cell 중심 Detection이었다면, 이제는 Anchor 중심 Detecion을 수행한다.
  * K-Means Clustering 을 통해 데이터 세트의 GT의 크기와 Ratio에 대해서 군집화 분류를 하고, 학습에 사용할 Anchor Box의 크기와 Ratio를 선정했다.
  * 위의 사진을 보면 K-Means Clustering의 결과로 5개 색깔의 묶음이 나왔다. 1묶음이 1개의 Anchor Box의 평균사이즈가 되어 총 5개의 Anchor box를 Yolo2에서 사용한다.

---

> Output

![이미지](https://user-images.githubusercontent.com/122156509/267208587-a513b004-0ddb-4824-974a-48cce61557af.png)
* 13 x 13 의 최종 Feature Map에는 5개의 Anchor Box에 대한 정보가 들어가 있다.
* 1개의 Anchor box에 대해서 25개의 정보를 가진다. 4(bounding box regression) + 1(Objectness Score x IOU)) + 20(class softmax)
