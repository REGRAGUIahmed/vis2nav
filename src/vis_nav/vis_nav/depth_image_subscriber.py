import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthImageSaver(Node):
    def __init__(self):
        super().__init__('depth_image_saver')
        self.i = 0
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Replace with your depth image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        # self.get_logger().info('Receiving depth image')
        try:
            # Convert ROS Image message to OpenCV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Normalize the depth image to fall within the 8-bit range
            cv_image_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            cv_image_normalized = cv_image_normalized.astype(np.uint8)

            # Save the normalized image as a .png file
            if self.i < 2:
                cv2.imwrite(f'/home/regmed/dregmed/vis_to_nav/src/vis_nav/vis_nav/results/depth_image_{self.i}.png', cv_image_normalized)
                self.get_logger().info(f'Depth image saved as depth_image_{self.i}.png')
                self.i += 1
        except Exception as e:
            self.get_logger().error('Could not convert depth image: %s' % str(e))

def main(args=None):
    rclpy.init(args=args)
    node = DepthImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

