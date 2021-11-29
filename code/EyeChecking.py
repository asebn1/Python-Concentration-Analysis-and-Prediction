from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # 두 점의 유클리드 거리 계산
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 수평 거리 유클리드 계산
    C = dist.euclidean(eye[0], eye[3])
    # 비율 계산 -> 눈 감는지 여부
    ear = (A + B) / (2.0 * C)
    
    return ear