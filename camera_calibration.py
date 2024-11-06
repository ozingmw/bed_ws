import cv2
import numpy as np

R_inv = None
t_inv = None
X_c_value = None
Y_c_value = None

def initialize(color_image, camera_info, marker_length):
    global R_inv, t_inv, X_c_value, Y_c_value

    d = np.zeros((4,1))
    k = np.array(camera_info.k).reshape((3,3))
    
    # ArUco 마커 탐지 및 외부 파라미터 계산
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        # 첫 번째 마커 사용
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, k, d)
        rvec = rvecs[0]
        tvec = tvecs[0].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T  # 역회전 행렬
        t_inv = -R_inv @ tvec

        height, width, _ = color_image.shape
        u = np.arange(width)
        v = np.arange(height)
        u_grid, v_grid = np.meshgrid(u, v)

        X_c_value = (u_grid - k[0, 2]) / k[0, 0]
        Y_c_value = (v_grid - k[1, 2]) / k[1, 1]
        return True
    return False

def camera2world(depth_image, roi):
    global R_inv, t_inv, X_c_value, Y_c_value

    x, y, w, h = roi

    # depth_image = depth_image.astype(np.float32)
    # k = k.astype(np.float32)
    # R_inv = R_inv.astype(np.float32)
    # t_inv = t_inv.astype(np.float32)
    # u_grid = u_grid.astype(np.float32)
    # v_grid = v_grid.astype(np.float32)
    
    # 깊이 이미지 (mm 단위)를 m 단위로 변환
    # depth = depth_image.astype(np.float32) / 1000.0  # shape: (height, width)

    # u, v 좌표 배열 생성

    # u_grid와 v_grid의 shape는 (height, width)

    # 카메라 좌표계에서의 X_c, Y_c 계산
    X_c = X_c_value[y:y+h, x:x+w] * depth_image[y:y+h, x:x+w]  # shape: (height, width)
    Y_c = Y_c_value[y:y+h, x:x+w] * depth_image[y:y+h, x:x+w]  # shape: (height, width)
    Z_c = depth_image[y:y+h, x:x+w]  # 이미 depth가 있음

    # 카메라 좌표 포인트들의 배열 생성
    # camera_points = np.stack((X_c, Y_c, Z_c), axis=-1)

    # 월드 좌표계에서의 Z 값 계산
    # Z_w = R_inv[2, :] @ [X_c, Y_c, Z_c] + t_inv[2]
    # 이를 전체 이미지에 대해 벡터화하여 계산

    # Z_w = R_inv[2, 0]*X_c + R_inv[2, 1]*Y_c + R_inv[2, 2]*Z_c + t_inv[2]
    Z_w = (R_inv[2, 0] * X_c) + (R_inv[2, 1] * Y_c) + (R_inv[2, 2] * Z_c) + t_inv[2]
    # Z_w의 shape는 (height, width)

    # 유효하지 않은 깊이 값 처리 (예: depth_image == 0)
    Z_w[depth_image[y:y+h, x:x+w] == 0] = np.nan  # 또는 다른 값을 지정하여 유효하지 않음을 표시

    Z_w = -Z_w

    return Z_w  # 전체 이미지에 대한 월드 좌표계의 높이(Z 값) 반환