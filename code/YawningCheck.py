from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # 두 점 사이 거리 계산 (수직)
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # 두 점 사이 거리 계산 (수평)
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

    # 계산
    yawnCheck = (A + B) / (2.0 * C)

    return yawnCheck