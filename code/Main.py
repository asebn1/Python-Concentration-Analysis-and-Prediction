
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
from cv2 import cv2
import numpy as np
from EyeChecking import eye_aspect_ratio
from YawningCheck import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
# 값 저장 ['time', 'concentration']
import pandas as pd
# 선형회귀
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False ## 마이나스 '-' 표시 제대로 출력
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

# 1. dlib의 face detector 초기화
detector = dlib.get_frontal_face_detector()
# 2. facial landmark predictor
print("[INFO] landmark predictor - loading ")
predictor = dlib.shape_predictor('./dlib/face_landmarks.dat')

# 카메라 세팅
print("[INFO] 카메라 초기화")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# 카메라 설정
frame_width = 1024
frame_height = 576

# 시간에 따른 점수 설정
start_time = time.time()
check_time = 0  # 시간차이
distTimeValue = 5 # 집중도를 5마다
score = 100 # 집중도
current_Score = 0 # 현재집중도
temp_score = score # 집중도 출력용
total_score = 0 # 전체 점수
eyeCheck = False
yawnCheck = False
headCheck = False
# 가장 작은 구간
minScope = 0
minScore = 100

# 데이터 저장 리스트
db = []

# 2D image points.
image_points = np.array([
    (359, 391),     # 코 끝 34
    (399, 561),     # 턱 9
    (337, 297),     # Left eye 37
    (513, 301),     # Right eye 46
    (345, 465),     # Left Mouth 49
    (453, 469)      # Right mouth 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.8
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0   # 눈깜빡임 카운터

# mouth의 랜드마크
(mStart, mEnd) = (49, 68)

a = 0
while True:
    # 크기 조절
    frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # 얼굴감지
    rects = detector(gray, 0)

    # 얼굴 검출되는지 확인
    # 얼굴의 프레임 수
    # if len(rects) > 0:
        # text = "{} face(s) found".format(len(rects))
        # cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # loop face detections
    for rect in rects:
        # 얼굴 경계 상자 -> 그리기
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        # 얼굴 랜드마크 결정
        # (x, y) 좌표를 numpy로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    # 입
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        mouthHull = cv2.convexHull(mouth)

        # 입 -> 그리기
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        # 하품 점수 추출
        # cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 하품시 출력
        if mar > MOUTH_AR_THRESH:
            # cv2.putText(frame, "Yawning!", (800, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 집중도 
            yawnCheck = True
        
    # -- 하품 출력 --
        text = "Yawning : "
        cv2.putText(frame, text, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if yawnCheck == False:
            text = "T"
            cv2.putText(frame, text, (270, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            text = "F"
            cv2.putText(frame, text, (270, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 그리기
        # 얼굴 랜드마크
        # 눈, 입만 색 다르게 출력
        for (i, (x, y)) in enumerate(shape):
            # 키 목록에 저장
            # keypoints = [(i,(x,y))]
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                # Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                # Red
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

        # 고개 위치 
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            # 고개변화 출력
            # cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 집중도
            if head_tilt_degree[0] < 14:
                headCheck = True
                
    # -- 고개위치 출력 --
        # text = "headPose : "
        # cv2.putText(frame, text, (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # if headCheck == False:
        #     text = "T"
        #     cv2.putText(frame, text, (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # else:
        #     text = "F"
        #     cv2.putText(frame, text, (490, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        
    # --  눈  --
        # 왼쪽 눈, 오른 쪽 눈 추출
        # 양쪽 눈 가로/세로 비율 계산

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 눈 가로/세로 비율의 평균
        ear = (leftEAR + rightEAR) / 2.0

        # 계산된 눈 -> 그리기
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        
    # -- 눈 깜빡임 --
        # 깜빡임 프레임 카운터 증가
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # 눈을 감고있는 경우 경고
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # cv2.putText(frame, "Eyes Closed!", (500, 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 눈 감음
                eyeCheck = True
            # 아닐 경우 초기화
        else:
            COUNTER = 0
            eyeCheck = False
        
    # -- 눈 출력 --
        text = "Eyes : "
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if eyeCheck == False:
            text = "T"
            cv2.putText(frame, text, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            text = "F"
            cv2.putText(frame, text, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        end_time = time.time()
        distance_time = end_time - start_time
        # print(distance_time)
    # 5초마다의 집중력 출력
        if distance_time > check_time:
            if eyeCheck == True:
                score = 0
            else:
                # 하품시 50
                if yawnCheck == True:
                    score -= 50
                # 시선 돌림
                if headCheck == True:
                    score -= 25
            # 5초마다
            check_time += 1
            if check_time % 5 == 0:
                distTimeValue += 5
                current_Score = temp_score//5
                temp_score = 0
                # 가장 작은구간 구하기
                if minScore > current_Score:
                    minScore = current_Score
                    minScope = check_time
            # db.append([distTimeValue, total_score]) : 선형
            db.append([check_time, score])
            print(distTimeValue, score)
            temp_score += score
            total_score += score
            score = 100
            eyeCheck = False
            yawnCheck = False
            headCheck = False

    # -- 시간 출력 --
        text = "Time : {}".format(distTimeValue)
        cv2.putText(frame, text, (807, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # -- 집중도 출력 --
        text = "Current Concentration : {}".format(current_Score)
        cv2.putText(frame, text, (615, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # frameq 출력
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키로 끝내기
    if key == ord("q"):
        break
# 종료
df = pd.DataFrame(db, columns=['time', 'concentration'])
df.to_csv('concentration_time.csv', index=False, encoding='cp949') 
cv2.destroyAllWindows()
vs.stop()

# 선형회귀 시작
df = pd.read_csv('./concentration_time.csv')
fit = ols('concentration ~ time',data=df).fit() ## 단순선형회귀모형 적합
## 시각화
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
 
font_size = 15
plt.scatter(df['time'],df['concentration']) ## 원 데이터 산포도

plt.xlabel('time', fontsize=font_size)
plt.ylabel('concentration',fontsize=font_size)
plt.plot(df['time'],fit.fittedvalues,color='red') ## 회귀직선 추가


# 강조 구간
span_start = minScope-4
span_end = minScope
plt.axvspan(span_start, span_end, facecolor='gray', alpha=0.5)

plt.show()
