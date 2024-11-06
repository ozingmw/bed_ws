import cv2
import numpy as np
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor

import rclpy
import cv_bridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from models import Azure, OpenAI
from cv_opti_cali import *


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        os.makedirs('./res/test1', exist_ok=True)
        os.makedirs('./res/test2', exist_ok=True)

        self.sub_camera_color_raw = self.create_subscription(Image, '/camera/color/image_raw', self.cb_color_raw, 10)
        self.sub_camera_depth_raw = self.create_subscription(Image, '/camera/depth/image_raw', self.cb_depth_raw, 10)
        # self.timer_position_check = self.create_timer(10*60, self.cb_timer_position_check)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.cb_camera_info, 10)

        self.cv2_bridge = cv_bridge.CvBridge()
        self.model = Azure()

        self.logger = self.get_logger()

        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.calibrated = False
        self.camera_checked = False

        HZ = 30

        self.roi = (230, 110, 1000-230, 530-110)
        self.alert_threshold_ratio_sit = None
        self.alert_threshold_ratio_none = None
        self.human_width = 500  # mm
        self.hist_bin_size = 100  # mm
        self.ratio_sit_margin = 0.02
        self.ratio_none_margin = 0.02

        self.marker_length = 0.185

        self.show_status_until = None
        self.show_position_until = None
        self.display_duration = 2  # 초

        # 버튼 설정
        self.button_test_1 = (20, 20, 80, 50)
        self.button_test_2 = (120, 20, 80, 50)
        self.button_re_cali = (220, 20, 80, 50)

        self.window_name = "test"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

        # 스레드 세이프를 위한 락
        self.lock = threading.Lock()

        # 스레드풀 초기화
        self.multi_executor = ThreadPoolExecutor(max_workers=2)
        self.checking_thread = []

        self.timer = self.create_timer(round(1/HZ,3), self.process_images)  # 0.1초마다 실행

    def save_log(self, path, text, is_text=True, is_image=True):
        now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
        if is_text:
            with open(f'{path}/result.txt', 'a') as f:
                f.write(f'{now}\t{text}\n')
        if is_image:
            cv2.imwrite(f'{path}/{now}.png', self.color_image)

        self.logger.info("LOGGING SUCCESS")

    def cb_color_raw(self, msg):
        try:
            color_image = self.cv2_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.color_image = color_image
        except cv_bridge.CvBridgeError as e:
            self.logger.error(f"CV Bridge Error in cb_color_raw: {e}")

    def cb_depth_raw(self, msg):
        try:
            depth_image = self.cv2_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.depth_image = depth_image
        except cv_bridge.CvBridgeError as e:
            self.logger.error(f"CV Bridge Error in cb_depth_raw: {e}")

    def cb_camera_info(self, msg):
        with self.lock:
            self.camera_info = msg

    def calibrate_threshold(self, depth_image):
        x, y, w, h = self.roi
        depth_rw_image = camera2world(depth_image)

        roi_image = depth_rw_image[y:y+h, x:x+w]

        # 히스토그램 계산
        max_depth_value = np.nanmax(roi_image)
        if np.isnan(max_depth_value):
            self.logger.error("Invalid depth values in ROI")
            return

        hist, bin_edges = np.histogram(
            roi_image[~np.isnan(roi_image)], bins=np.arange(1, max_depth_value + self.hist_bin_size, self.hist_bin_size)
        )

        # 가장 빈도수가 높은 거리 구간 찾기
        max_bin_index = np.argmax(hist)
        dominant_range_start = bin_edges[max_bin_index]
        dominant_range_end = bin_edges[max_bin_index + 1]

        # 임계 거리 설정
        self.threshold_distance_none = (dominant_range_start + dominant_range_end) / 2
        self.threshold_distance_none = self.threshold_distance_none - 50
        self.threshold_distance_none = max(self.threshold_distance_none, 100)

        self.threshold_distance_sit = self.threshold_distance_none - self.human_width

        self.logger.info(f"Threshold distance calibrated to: (none: {self.threshold_distance_none:.2f} mm, sit: {self.threshold_distance_sit:.2f} mm)")

        # ROI 내 임계 거리보다 가까운 픽셀 비율 계산 (sit)
        mask_sit = (roi_image < self.threshold_distance_sit) & (~np.isnan(roi_image))
        total_pixels = np.count_nonzero(~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_sit)
        ratio_sit = masked_pixels / total_pixels if total_pixels > 0 else 0
        self.alert_threshold_ratio_sit = ratio_sit + self.ratio_sit_margin
        self.alert_threshold_ratio_sit = min(self.alert_threshold_ratio_sit, 1.0)
        self.logger.info(f"Alert threshold ratio(sit) set to: {self.alert_threshold_ratio_sit:.2f}")

        # ROI 내 임계 거리보다 가까운 픽셀 비율 계산 (none)
        mask_none = (roi_image >= self.threshold_distance_sit) & (roi_image <= self.threshold_distance_none) & (~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_none)
        ratio_none = masked_pixels / total_pixels if total_pixels > 0 else 0
        self.alert_threshold_ratio_none = ratio_none + self.ratio_none_margin
        self.alert_threshold_ratio_none = min(self.alert_threshold_ratio_none, 1.0)
        self.logger.info(f"Alert threshold(none) ratio set to: {self.alert_threshold_ratio_none:.2f}")

    def check_threshold_in_roi(self, depth_image):
        x, y, w, h = self.roi
        depth_rw_image = camera2world(depth_image)

        roi_image = depth_rw_image[y:y+h, x:x+w]

        mask_sit = (roi_image < self.threshold_distance_sit) & (~np.isnan(roi_image))
        total_pixels = np.count_nonzero(~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_sit)
        ratio_sit = masked_pixels / total_pixels if total_pixels > 0 else 0
        full_mask_sit = np.zeros_like(depth_rw_image, dtype=bool)
        full_mask_sit[y:y+h, x:x+w] = mask_sit

        mask_none = (roi_image >= self.threshold_distance_sit) & (roi_image <= self.threshold_distance_none) & (~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_none)
        ratio_none = masked_pixels / total_pixels if total_pixels > 0 else 0
        full_mask_none = np.zeros_like(depth_rw_image, dtype=bool)
        full_mask_none[y:y+h, x:x+w] = mask_none

        return full_mask_sit, ratio_sit, full_mask_none, ratio_none

    def process_images(self):
        with self.lock:
            if self.color_image is None or self.depth_image is None or self.camera_info is None:
                return
            
            # 이미지와 카메라 정보를 복사하여 락 해제 후 처리
            color_image = self.color_image.copy()
            depth_image = self.depth_image.copy()
            camera_info = self.camera_info

        # 카메라 파라미터 초기화
        if not self.camera_checked:
            if initialize(color_image, camera_info, self.marker_length):
                self.camera_checked = True
            else:
                return

        # 임계값 캘리브레이션
        if not self.calibrated:
            self.calibrate_threshold(depth_image)
            self.calibrated = True
            return
        
        self.visualize_image(color_image)

    def visualize_image(self, color_image):
        vis_image = color_image.copy()

        x, y, w, h = self.roi
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.draw_button(vis_image, self.button_test_1, "TEST1")
        self.draw_button(vis_image, self.button_test_2, "TEST2")
        self.draw_button(vis_image, self.button_re_cali, "RE CALI")

        if self.show_status_until and time.time() < self.show_status_until:
            text_size, _ = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            text_x = vis_image.shape[1] - text_size[0] - 10
            text_y = vis_image.shape[0] - 10
            cv2.putText(
                vis_image, f"{self.status_text}", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, self.status_color, 2
            )
        else:
            self.show_status_until = None

        if self.checking_thread and self.checking_thread[0].done():
            if self.show_position_until and time.time() < self.show_position_until:
                text_size, _ = cv2.getTextSize(self.position_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                text_x = vis_image.shape[1] - text_size[0] - 10
                text_y = vis_image.shape[0] - 10
                cv2.putText(
                    vis_image, f"{self.position_text}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, self.status_color, 2
                )
            elif self.show_position_until and time.time() >= self.show_position_until:
                self.checking_thread.pop(0)
                self.show_position_until = None
            else:
                self.show_position_until = time.time() + self.display_duration

        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)

    def check_current_status(self, ratio_sit, ratio_none):
        if ratio_sit > self.alert_threshold_ratio_sit:
            self.status_text = "sit"
            self.status_color = (0, 0, 255)  # 빨간색
        elif ratio_none > self.alert_threshold_ratio_none:
            self.status_text = "lay"
            self.status_color = (255, 0, 0)  # 파란색
        else:
            self.status_text = "none"
            self.status_color = (0, 255, 0)  # 초록색

    def check_test(self, depth_image):
        full_mask_sit, ratio_sit, full_mask_none, ratio_none = self.check_threshold_in_roi(depth_image)

        self.check_current_status(ratio_sit, ratio_none)

    def is_point_in_button(self, x, y, button):
        bx, by, bw, bh = button
        return (bx <= x <= bx + bw) and (by <= y <= by + bh)

    def draw_button(self, image, button_data, text):
        x, y, w, h = button_data
        cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), -1)
        cv2.putText(image, text, (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_image = self.depth_image.copy()
            if self.is_point_in_button(x, y, self.button_test_1):
                self.check_test(depth_image)
                self.show_status_until = time.time() + self.display_duration
                if self.status_text == 'sit':
                    self.save_log(f'./res/test1', self.status_text)
                elif self.status_text == 'lay':
                    color_image = self.color_image.copy()
                    self.checking_thread.append(self.multi_executor.submit(self.run_model_generate, color_image, 'test1'))
                else:
                    self.logger.info("nobody detected")
            elif self.is_point_in_button(x, y, self.button_test_2):
                self.check_test(depth_image)
                if self.status_text == 'sit':
                    self.save_log(f'./res/test2', self.status_text)
                elif self.status_text == 'none':
                    self.save_log(f'./res/test2', self.status_text)
                else:
                    self.logger.info("body lay detected")
            elif self.is_point_in_button(x, y, self.button_re_cali):
                self.calibrate_threshold(depth_image)
            else:
                depth_rw_image = camera2world(depth_image.copy())
                if depth_rw_image is not None:
                    self.logger.info(f"{depth_rw_image[y, x]}")
                else:
                    self.logger.info("Failed to get world coordinate depth")

    def run_model_generate(self, color_image, test):
        response = self.model.generate(color_image)

        _convert_k2e = {
            '정면': 'front',
            '측면': 'side'
        }
        self.position_text = _convert_k2e.get(response, 'unknown')
        if test:
            cv2.imwrite(f'./res/{test}', color_image)
            self.save_log(f'./res/{test}', self.position_text, is_image=False)

    def stop(self):
        self.multi_executor.shutdown(wait=False)


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Camera node stopped cleanly')
    finally:
        camera_node.stop()
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()