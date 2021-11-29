
### 주제
- 비대면 학습 집중도 분석 및 예측
- Analysis and prediction of the concentration of non-face-to-face learning

---
# Table of Contents
* [About the Project](#about-the-project)
* [Building](#Building)
* [Run Screen](#Run-Screen)
* [Member](#Member)

# About The Project
    COVID-19로 인해 원격 수업은 지금까지 이어지고 있다. 특히 지난해 학생들의 학업성취도가 크게 떨어진 것으로 나타났다. 중상위권 비율은 줄었으며 기초학력 미달 학생들이 늘었다. 고2 수학 기초학력은 2019년에 9%에서 2020년에 13.5%로 무려 50%가 늘었다. 이러한 관점에서 비대면 학습의 문제점과 해결 방안에 관심이 증대되고 있다. 이러한 관점에서 학습자의 편의성과 학습 증대를 위해 졸음 감지 알고리즘은 다양한 방법으로 연구되고 있다. 크게 3가지 형태로 분류된다. 첫째, 학습자의 생체적 특성을 분석하는 방법으로 뇌파, 심장박동, 맥박 수 등을 측정하여 졸음 및 집중도 여부의 정확도가 높지만, 인체에 직접적으로 특별한 장치를 부착하는 접촉방식이기 때문에 실용적이지 못하다. 둘째, 컴퓨터 비전 기술을 이용하여 학습자의 특징 변화를 측정함으로써 졸음 여부를 판단하는 방법이다. 학습자의 얼굴에는 많은 변화가 일어난다. 피로 하거나 주의력이 감소된 사람은 눈을 감거나 아주 작게 뜨며 고개의 숙임, 하품 등의 얼굴 특징으로 쉽게 구별할 수 있다. 이러한 얼굴 특징 변화를 관찰하여 학습자의 졸음 상태를 판단할 수 있는 비접촉식 방법이므로, 컴퓨터 비전 기술을 이용한 방법은 학습자에게 적용이 가능하다. 이러한 점을 토대로 비대면 학습에 적용 가능한 집중도 분석을 구현하고자 한다.

# Building
1. installing 1
```
git clone https://github.com/asebn1/Python-Concentration-Analysis-and-Prediction.git
```
2. installing 2
```
pip install -r Requirements.txt
```
3. Run executable
```
python3 Social_Distance_avg_run.py
```

# Member
**Project Member**
- 공재호([asebn1](https://github.com/asebn1))

**Project Mentor**
- 배성호
