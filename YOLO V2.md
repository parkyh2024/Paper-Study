## YOLO Version 2

> 핵심 특징
  * Grid Cell 당 Anchor Box 수 : 5
  * Anchor box 사용 - 결정 방법 : K-means Clustering. Anchor box 크기와 ratio를 정하는 방법. GT Box를 그룹핑 한 후, 그것을 이용해서 Anchor box 크기와 ratio 결정.
  * Output Peature Map 크기 : 13 x 13
  * Batch Normaliztion 을 적극적으로 활용하기 시작
  * High Resolution Classifier : 처음에 Input 224 x 224로 학습하다가, input 448 x 448로 Fine tuning 하여 성능향상을 얻음.
  * 서로 다른 크기의 Image들 섞어서 배치하여 작은 Object도 Detect할 수 있도록 노력.
![이미지]([https://github.com/parkyh2024/Paper-Study/assets/122156509/96f29df4-9ef5-4505-b832-815132c8bcac](https://user-images.githubusercontent.com/122156509/267206109-96f29df4-9ef5-4505-b832-815132c8bcac.png)https://user-images.githubusercontent.com/122156509/267206109-96f29df4-9ef5-4505-b832-815132c8bcac.png)
