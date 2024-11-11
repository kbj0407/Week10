import cv2
import numpy as np

# HSV 범위 조정
yellow_lower = np.array([10, 80, 100])    # 채도 범위 확장
yellow_upper = np.array([40, 255, 255])
white_lower = np.array([0, 0, 160])       # Value 하한값 낮추기
white_upper = np.array([180, 50, 255])

# 라즈베리 파이 카메라 사용 (0번 장치)
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break
    
    # 프레임 크기 조정 (640x480)
    frame_resized = cv2.resize(frame, (640, 480))

    # 대비 조정
    lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)               # 밝기 채널에 히스토그램 평활화 적용
    lab = cv2.merge((l, a, b))
    frame_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # BGR -> HSV 변환
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

    # 노란색과 흰색 마스크 생성
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    # 두 마스크 결합
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    # 마스크를 원본 이미지에 적용
    result = cv2.bitwise_and(frame_resized, frame_resized, mask=combined_mask)

    # 배경을 검게 하여 선만 남기기
    background = np.zeros_like(frame_resized)
    result_with_black_bg = np.where(result > 0, result, background)

    # 결과를 화면에 표시
    cv2.imshow('Yellow and White Line Detection', result_with_black_bg)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
