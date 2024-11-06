import cv2
import numpy as np
import threading
import os
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageFont, ImageDraw, Image as PILImage

import rclpy
import cv_bridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer

from models import Azure
from camera_calibration import *


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        os.makedirs('./res/test1', exist_ok=True)
        os.makedirs('./res/test2', exist_ok=True)

        sub_camera_color_raw = Subscriber(self, Image, '/camera/color/image_raw')
        sub_camera_depth_raw = Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = ApproximateTimeSynchronizer(
            [sub_camera_color_raw, sub_camera_depth_raw], 
            queue_size=10, 
            slop=0.1
        )
        
        self.ts.registerCallback(self.process_images)
        self.sub_camera_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.cb_camera_info, 10)

        self.cv2_bridge = cv_bridge.CvBridge()
        self.logger = self.get_logger()

        self.model = Azure()

        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.calibrated = False
        self.camera_checked = False

        # aruco marker 크기
        self.marker_length = 0.2    # m

        # GUI를 위한 설정
        self.roi = (230, 100, 1280-230*2, 720-110*2)
        self.alert_threshold_ratio_sit = None
        self.alert_threshold_ratio_none = None
        self.human_width = 400  # mm
        self.hist_bin_size = 100  # mm
        self.ratio_sit_margin = 0.03
        self.ratio_none_margin = 0.10
        self.show_status_until = None
        self.show_position_until = None
        self.display_duration = 2  # 초

        # 버튼 설정
        self.button_test_1 = (20, 20, 80, 50)
        self.button_test_2 = (120, 20, 80, 50)
        self.button_re_cali = (220, 20, 80, 50)

        self.window_name = "Pose & Behavior Recognizer"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

        # 스레드 세이프를 위한 락
        self.lock = threading.Lock()

        # 스레드풀 초기화
        self.multi_executor = ThreadPoolExecutor(max_workers=2)
        self.checking_thread = []

        fontpath = "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"
        self.font = ImageFont.truetype(fontpath, 120)
        self.e2k_text = {
            'sit': ['앉기', '기상'],
            'lying': ['정면or측면', '기상'],
            'none': ['', '이탈']
        }

    def save_log(self, path, text, image):
        now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
        if text is not None:
            with open(f'{path}/result.txt', 'a', encoding='utf-8') as f:
                f.write(f'{now}\t{text}\n')
        if image is not None:
            cv2.imwrite(f'{path}/{now}.png', image)

    def cb_camera_info(self, msg):
        with self.lock:
            self.camera_info = msg

    def process_images(self, color_image, depth_image):
        try:
            color_image = self.cv2_bridge.imgmsg_to_cv2(color_image, desired_encoding='bgr8')
            with self.lock:
                self.color_image = color_image
        except cv_bridge.CvBridgeError as e:
            self.logger.error(f"CV Bridge Error in cb_color_raw: {e}")
        try:
            depth_image = self.cv2_bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
            with self.lock:
                self.depth_image = depth_image
        except cv_bridge.CvBridgeError as e:
            self.logger.error(f"CV Bridge Error in cb_depth_raw: {e}")

        if self.camera_info is None:
            return
        
        with self.lock:
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

        # ROI 내 임계값 체크
        full_mask_sit, ratio_sit, full_mask_none, ratio_none = self.check_threshold_in_roi(depth_image)

        # 시각화
        self.visualize_image(full_mask_sit, ratio_sit, full_mask_none, ratio_none, color_image)

    def calibrate_threshold(self, depth_image):
        roi_image = camera2world(depth_image, self.roi)

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
        self.threshold_distance_none = self.threshold_distance_none - 100
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
        roi_image = camera2world(depth_image, self.roi)

        mask_sit = (roi_image < self.threshold_distance_sit) & (~np.isnan(roi_image))
        total_pixels = np.count_nonzero(~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_sit)
        ratio_sit = masked_pixels / total_pixels if total_pixels > 0 else 0
        full_mask_sit = np.zeros_like(depth_image, dtype=bool)
        full_mask_sit[y:y+h, x:x+w] = mask_sit

        mask_none = (roi_image >= self.threshold_distance_sit) & (roi_image <= self.threshold_distance_none) & (~np.isnan(roi_image))
        masked_pixels = np.count_nonzero(mask_none)
        ratio_none = masked_pixels / total_pixels if total_pixels > 0 else 0
        full_mask_none = np.zeros_like(depth_image, dtype=bool)
        full_mask_none[y:y+h, x:x+w] = mask_none

        return full_mask_sit, ratio_sit, full_mask_none, ratio_none

    def visualize_image(self, mask_sit, ratio_sit, mask_none, ratio_none, color_image):
        vis_image = color_image.copy()

        # 마스킹 시각화
        vis_image[mask_none] = [255, 0, 0]  # 파란색
        vis_image[mask_sit] = [0, 0, 255]  # 빨간색

        # ROI 사각형 그리기
        x, y, w, h = self.roi
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if ratio_sit > self.alert_threshold_ratio_sit:
            self.status_text = "sit"
            status_color = (0, 0, 255)  # 빨간색
        elif ratio_none > self.alert_threshold_ratio_none:
            self.status_text = "lying"
            status_color = (255, 0, 0)  # 파란색
        else:
            self.status_text = "none"
            status_color = (0, 255, 0)  # 초록색

        cv2.putText(
            vis_image, f"{self.status_text} | sit: {ratio_sit:.2f} | lying: {ratio_none:.2f}", (vis_image.shape[1] - 400, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        self.draw_button(vis_image, self.button_test_1, "TEST1")
        self.draw_button(vis_image, self.button_test_2, "TEST2")
        self.draw_button(vis_image, self.button_re_cali, "RE CALI")

        if self.show_status_until and time.time() < self.show_status_until:
            text_x = int(vis_image.shape[1]/2)
            text_y = vis_image.shape[0] - 130
            vis_image = PILImage.fromarray(vis_image)
            draw = ImageDraw.Draw(vis_image)
            draw.text((text_x, text_y), self.show_text, (0,0,255), font=self.font)
            vis_image = np.array(vis_image)
        else:
            self.show_status_until = None

        # Button 누를 때 이벤트 처리
        if self.checking_thread and self.checking_thread[0].done():
            if self.show_position_until and time.time() < self.show_position_until:
                self.show_text = self.position_text
                text_x = int(vis_image.shape[1]/2)
                text_y = vis_image.shape[0] - 130
                vis_image = PILImage.fromarray(vis_image)
                draw = ImageDraw.Draw(vis_image)
                draw.text((text_x, text_y), self.show_text, (0,0,255), font=self.font)
                vis_image = np.array(vis_image)
            elif self.show_position_until and time.time() >= self.show_position_until:
                self.checking_thread.pop(0)
                self.show_position_until = None
                self.position_text = ""
            else:
                self.show_position_until = time.time() + self.display_duration
        elif self.checking_thread and not self.checking_thread[0].done():
            self.show_text = '판독중'
            text_x = int(vis_image.shape[1]/2)
            text_y = vis_image.shape[0] - 130
            vis_image = PILImage.fromarray(vis_image)
            draw = ImageDraw.Draw(vis_image)
            draw.text((text_x, text_y), self.show_text, (0,0,255), font=self.font)
            vis_image = np.array(vis_image)

        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)

    def draw_button(self, image, button_data, text):
        x, y, w, h = button_data
        cv2.rectangle(image, (x, y), (x + w, y + h), (200, 200, 200), -1)
        cv2.putText(image, text, (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.lock:                
                color_image = self.color_image.copy()
                depth_image = self.depth_image.copy()

            if self.is_point_in_button(x, y, self.button_test_1):
                if self.status_text == 'sit':
                    self.show_text = self.e2k_text[self.status_text][0]
                    self.show_status_until = time.time() + self.display_duration
                    self.save_log(f'./res/test1', self.show_text, color_image)
                elif self.status_text == 'lying':
                    self.checking_thread.append(self.multi_executor.submit(self.run_model_generate, color_image))
                else:
                    self.logger.info("body not detected")
            elif self.is_point_in_button(x, y, self.button_test_2):
                self.show_text = self.e2k_text[self.status_text][1]
                self.show_status_until = time.time() + self.display_duration
                self.save_log(f'./res/test2', self.show_text, color_image)
            elif self.is_point_in_button(x, y, self.button_re_cali):
                self.calibrate_threshold(depth_image)
            # # DEBUG
            # else:
            #     depth_rw_image = camera2world(depth_image, self.roi)
            #     x1, y1, w1, h1 = self.roi
            #     depth_rw_image = np.nan_to_num(depth_rw_image, nan=0.0, posinf=0.0, neginf=0.0)
            #     depth_image[y1:y1+h1, x1:x1+w1] = depth_rw_image
            #     if depth_rw_image is not None:
            #         self.logger.info(f"{depth_image[y, x]}")
            #     else:
            #         self.logger.info("Failed to get world coordinate depth")

    def is_point_in_button(self, x, y, button):
        bx, by, bw, bh = button
        return (bx <= x <= bx + bw) and (by <= y <= by + bh)

    def run_model_generate(self, color_image):
        response = self.model.generate(color_image)

        self.position_text = response
        self.save_log(f'./res/test1', response, color_image)

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
