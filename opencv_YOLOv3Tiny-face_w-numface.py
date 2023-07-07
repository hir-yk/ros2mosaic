import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge


class MosaicNode(Node):
    def __init__(self):
        super().__init__('mosaic_node')
        self.publisher_ = self.create_publisher(Image, 'mosaic_image', 10)
        self.publisher_num_face_ = self.create_publisher(Int32, '/num_face', 10)
        self.subscription_ = self.create_subscription(Image, '/image_raw', self.mosaic_callback, 10)
        self.bridge = CvBridge()
        
        # YOLO関連の設定
        self.yolo_model = '/home/autoware/repo/ros2mozaic/models/face-yolov3-tiny.cfg'
        self.yolo_weights = '/home/autoware/repo/ros2mozaic/models/face-yolov3-tiny_41000.weights'
        self.confidence_threshold = 0.1
        self.net = cv2.dnn.readNetFromDarknet(self.yolo_model, self.yolo_weights)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def mosaic_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        (h, w) = img.shape[:2]

        # 画像を正規化するために、blobイメージを作成
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # YOLOにより顔検出を行う
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)

        faces_count = len(idxs)
        num_face_msg = Int32()
        num_face_msg.data = faces_count
        self.publisher_num_face_.publish(num_face_msg)

        if faces_count > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                margin_w = int(w * 0.1)
                margin_h = int(h * 0.0)
                startX, startY = max(x - margin_w, 0), max(y - margin_h, 0)
                endX, endY = min(x + w + margin_w, img.shape[1]), min(y + h + margin_h, img.shape[0])
                
                # モザイク処理
                mosaic_size = (max(int(w / (50)), 1), max(int(h / (50)), 1))
                face = img[startY:endY, startX:endX]
                small_face = cv2.resize(face, mosaic_size, interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(small_face, (endX - startX, endY - startY), interpolation=cv2.INTER_LINEAR)
                img[startY:endY, startX:endX] = mosaic

        # パブリッシュ
        img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.publisher_.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MosaicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
