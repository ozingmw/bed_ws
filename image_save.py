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
from test_cali import *


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        os.makedirs('./temp', exist_ok=True)

        self.sub_camera_color_raw = self.create_subscription(Image, '/camera/color/image_raw', self.cb_color_raw, 10)
        self.cv2_bridge = cv_bridge.CvBridge()
        self.count = 0
        cv2.namedWindow("test")
        cv2.setMouseCallback("test", self.on_mouse_click)

    def cb_color_raw(self, msg):
        self.msg = msg
        self.color_image = self.cv2_bridge.imgmsg_to_cv2(self.msg, desired_encoding='bgr8')

        cv2.imshow('test', self.color_image)
        cv2.waitKey(1)
        
    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count += 1
            cv2.imwrite(f'./temp/{self.count}.png', self.color_image)


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        camera_node.get_logger().info('Camera node stopped cleanly')
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
