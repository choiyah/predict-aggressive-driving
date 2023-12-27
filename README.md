# Unsupervised Driver Behavior Profiling leveraging Recurrent Neural Networks 
WISA 22nd (2021)에 발표한 논문

1. [모델 구조: Unsupervised Learning](#모델-구조:-Unsupervised-Learning)
2. [학습 데이터셋 구조](#학습-데이터셋-구조)
3. [학습 데이터셋 구조: Feature Engineering](#학습-데이터셋-구조-feature-engineering)
    - [Step 1. Timestamp Calibration](#step-1-timestamp-calibration)
    - [Step 2. Scaling](#step-2-scaling)
    - [Step 3. Window Sliding](#step-3-window-sliding)
4. [실험 결과](#실험-결과)
   
### 모델 구조: Unsupervised Learning
- 운전자 행동 프로파일링을 **이상 탐지** 개념으로 접근
- 타임 윈도우 피쳐 벡터를 인풋하여, **그 다음 타임 윈도우 피쳐 벡터를 예측**하는 **RNN** 모델 설계
  - 학습은 **정상 데이터로만** 이루어짐
- **일반적인 (정상) 운전 패턴** 시퀀스에서는 **낮은 regression (회귀) 에러**가, **그 외 운전 패턴 (난폭운전)** 시퀀스에서는 **높은 회귀 에러**가 산출
  <img width="1127" alt="image" src="https://github.com/choiyah/predict-aggressive-driving/assets/62586517/b557fd69-3e48-45e1-b0cd-9be81f70614f">

### 학습 데이터셋 구조
- ‘Driver Behavior Dataset’ ([github](https://github.com/jair-jr/driverBehaviorDataset))
- 각 장치 데이터들의 주파수를 맞추어 시계열 데이터로 재구성
  - 12 features: acceleration, linear acceleration, magnetometer, and gyroscope * x/y/z
    <img width="830" alt="image" src="https://github.com/choiyah/predict-aggressive-driving/assets/62586517/7982c327-ab16-4e4a-94e7-b5f47b51393c">
- 타임 윈도우 (일정 시간)만큼의 feature vector들을 **시퀀스**와 그 직후 **실측 feature vector를 페어**로 학습 데이터 생성
  ![image](https://github.com/choiyah/predict-aggressive-driving/assets/62586517/18b1a978-fa06-453d-9cce-e3847fa119f9)

### 학습 데이터셋 구조: Feature Engineering
#### Step 1. Timestamp Calibration
- feature (acceleration, magnetometer, gyroscope)들의 frequency를 맞춰줌
- **Downsampling**
  - frequency가 더 낮은 feature를 기준으로 잡고,  중복 주기에서의 **초기값**을 **대표값**으로 샘플링
    <img width="728" alt="image" src="https://github.com/choiyah/predict-aggressive-driving/assets/62586517/90f235f2-65fd-4f40-b8ff-96781ba01256">

#### Step 2. Scaling
- feature (acceleration, linear acceleration, magnetometer, gyroscope)들을 min-max scaling (0~1)
- **난폭 운전 데이터는 정상 운전 데이터를 기반으로 스케일링**

#### Step 3. Window Sliding
- window 크기를 정하여 1 row씩 슬라이딩
- window 크기의 feature vectors 시퀀스와 그 직후 feature vector를 실측 데이터로 묶음
  - (**Sequence of feature vectors**, **Ground-Truth of single feature vector**)
    - Sequence: Input to the model
    - Ground-Truth: for the model’s prediction
  <img width="464" alt="image" src="https://github.com/choiyah/predict-aggressive-driving/assets/62586517/6a155a5a-939b-4b74-b0a7-8f2125101e3a">

### 실험 결과
- 평가 방법: **ROC-AUC**
  - ‘난폭 운전’ 판단에서의 성공(TPR), 실패 (FPR) 여부를 나누는 기준을 확인할 수 있는 ROC 커브 선정
  - AUC (0.5~1.0)로부터 최고의 분류 성능을 측정
- 전체적으로 좋은 성능을 보였지만, Aggressive Acceleration을 식별하는 성능은 좋지 않음
<img width="1313" alt="image" src="https://github.com/choiyah/predict-aggressive-driving/assets/62586517/d0d586ff-20eb-4351-bdf4-ddd37971f2ab">

