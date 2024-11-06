# 침대 위 자세 인식

ROS2 & RGB-D 카메라를 이용한 현실 좌표 임계값 구분

---
### 실행 환경
ROS2 humble, 22.04 ubuntu

1\. prompt를 위한 루트 폴더에 정면, 측면 사진 필요
```
|
|ㅡ main.py
|ㅡ ...
|ㅡ prompt_image_정면.png
|ㅡ prompt_image_측면.png
```
2\. opencv gui 나눔명조 폰트 사용\
[링크](https://hangeul.naver.com/fonts/search?f=nanum)

---

### 실행 방법
- 필요 라이브러리 설치
```
pip install -r requirements.txt
```

- 실행
```
./launch.sh

or

python main.py
```

- GUI 개선판 (버튼 누를 당시만 좌표계산)
```
./launch.opti.sh
```