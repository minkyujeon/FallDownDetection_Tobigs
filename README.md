## Fall Detection

Tobig's Project (2018.06~2018.08)

Implemented a model to detect someone falling down when they are alone at home

Applied CNN+LSTM with self-generated video data

- 기존
  - Optical Flow에 CNN으로 이미지로 판별
  - sequence of frames contains a person falling
- 데이터

  - 인터넷에서 검색하여 긁어모았지만 데이터들이 부정확, 불분명함
  - 직접 만듦 - Fall down 5400개 / Non Fall down 5600개

  - fall down, sleep, sit, walk, bend
- 모델 선정 및 설명

  - 동영상을 사진의 frame으로 바꿔준 후 사진을 처리할 수 있는 '과정'이 필요
  - 넘어지는 과정, 흐름을 기억해야 하므로 RNN(LSTM)이 필요
- 모델 1(VGG 16 with fine tuning + LSTM)

  - 다른 논문에서 pretrained로 VGG16을 사용하는 것을 봄 - 쉽게 사용가능
  - 이미지를 처리해서 분류하는 문제이므로 spatial한 정보를 처리하기 위해 기본적으로 CNN을 생각
  - 데이터 부족을 극복하기 위해 fine tuning 기법을 사용
  - 넘어지는 sequential한 상황을 인지해야 하므로 LSTM과 접목
- 모델2(Skeleton + LSTM)

  - 모델1에서 CNN으로 사진에서 사람의 feature를 따내는 것은 사람 뿐만 아니라 다른 물체들도 따낼 수 있음
  - 사람에 더 집중할 수 있는 방법을 찾아보자!
  - 사람 몸의 골격에 집중하여 feature를 뽑아내면 더 사람의 행동에 집중하여 따낼 수 있지 않을까
- 성능 평가
  - pd.crosstab()
  - 교차분석

#### 선행연구

- conv 5층 + fc nn 3층 -> softmax로 예측