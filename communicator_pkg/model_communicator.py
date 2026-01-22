import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import base64
import json
import os
import requests
import threading
from typing import List, Optional, Tuple


class ModelCommunicator(Node):
    def __init__(self):
        super().__init__('communicator_node')
        self.declare_parameter('image_topic', '/image_rect/compressed')
        self.declare_parameter('model_api_url', 'https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host')
        self.declare_parameter('prompt', 'Move to the monitor, stop in front of it, and avoid any collisions')
        self.declare_parameter('model_api_health_check', True)
        self.declare_parameter('model_api_timeout_sec', 600)
        self.declare_parameter('result_save_dir', '')
        self.declare_parameter('publish_path_topic', '/waypoints/path')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('exit_after_request', False)
        self._image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self._model_api_url = self.get_parameter('model_api_url').get_parameter_value().string_value
        self._prompt = (
            self.get_parameter('prompt').get_parameter_value().string_value
        )
        self._model_api_health_check = (
            self.get_parameter('model_api_health_check').get_parameter_value().bool_value
        )
        self._model_api_timeout_sec = (
            self.get_parameter('model_api_timeout_sec').get_parameter_value().integer_value
        )
        self._result_save_dir = (
            self.get_parameter('result_save_dir').get_parameter_value().string_value
        )
        self._publish_path_topic = (
            self.get_parameter('publish_path_topic').get_parameter_value().string_value
        )
        self._publish_rate_hz = (
            self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        )
        self._frame_id = (
            self.get_parameter('frame_id').get_parameter_value().string_value
        )
        self._exit_after_request = (
            self.get_parameter('exit_after_request').get_parameter_value().bool_value
        )
        if not self._result_save_dir:
            self._result_save_dir = os.path.join(self._get_workspace_root(), 'result')
        self._received_first = False
        self._done = False
        self._trajectory_lock = threading.Lock()
        self._trajectory_2d: Optional[List[Tuple[float, float]]] = None
        self._subscription = self.create_subscription(
            CompressedImage,
            self._image_topic,
            self._on_compressed_image,
            qos_profile_sensor_data,
        )
        self._path_pub = self.create_publisher(Path, self._publish_path_topic, 10)
        self._publish_timer = self.create_timer(
            1.0 / max(self._publish_rate_hz, 0.1),
            self._on_publish_timer,
        )
        self.get_logger().info(f'Subscribed to image topic: {self._image_topic}')
        self.get_logger().info(f'Publishing path to: {self._publish_path_topic}')

    def _on_compressed_image(self, msg: CompressedImage):
        if self._received_first:
            return
        self._received_first = True
        self.get_logger().info('Received CompressedImage message, sending to server.')
        # Stop receiving additional images to avoid redundant callbacks.
        self.destroy_subscription(self._subscription)
        thread = threading.Thread(
            target=self._call_model_api,
            args=(msg.data,),
            daemon=True,
        )
        thread.start()

    @property
    def done(self):
        return self._done

    def _get_workspace_root(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(3):
            current_dir = os.path.dirname(current_dir)
        return current_dir

    def _call_model_api(self, image_bytes: bytes):
        b64_str = base64.b64encode(image_bytes).decode('utf-8')
        payload = {
            'prompt': self._prompt,
            'image_base64': b64_str,
        }
        headers = {
            'Content-Type': 'application/json',
        }
        try:
            if self._model_api_health_check:
                health = requests.get(
                    f'{self._model_api_url}/healthz',
                    timeout=30,
                )
                health.raise_for_status()
                self.get_logger().info(f'healthz: {health.json()}')
            self.get_logger().info(f'Sending request to model API: {self._model_api_url}/infer_waypoints')
            resp = requests.post(
                f'{self._model_api_url}/infer_waypoints',
                headers=headers,
                data=json.dumps(payload),
                timeout=self._model_api_timeout_sec,
            )
            resp.raise_for_status()
            data = resp.json()
            trajectory_2d = data.get('trajectory_2d')
            if isinstance(trajectory_2d, list):
                with self._trajectory_lock:
                    self._trajectory_2d = [
                        (float(x), float(y)) for x, y in trajectory_2d
                    ]
            self.get_logger().info(f'trajectory_2d: {trajectory_2d}')
            request_id = data.get('request_id')
            if request_id is not None:
                self.get_logger().info(f'request_id: {request_id}')
            video_path = data.get('video_path')
            if video_path is not None:
                self.get_logger().info(f'video_path: {video_path}')
            if request_id:
                self._download_result_json(request_id)
                self._download_mp4(request_id)
        except requests.RequestException as exc:
            self.get_logger().error(f'Failed to call model API: {exc}')
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f'Invalid response from model API: {exc}')
        self.get_logger().info('Request finished.')
        if self._exit_after_request:
            self.get_logger().info('exit_after_request=True, shutting down.')
            self.destroy_subscription(self._subscription)
            self._done = True

    def _on_publish_timer(self):
        with self._trajectory_lock:
            points = list(self._trajectory_2d) if self._trajectory_2d else None
        if not points:
            return
        path = Path()
        path.header.frame_id = self._frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header.frame_id = self._frame_id
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = y
            pose.pose.position.y = x
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

    def _download_result_json(self, request_id: str):
        try:
            url = f'{self._model_api_url}/requests/{request_id}/result.json'
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            result_json_data = resp.json()
            os.makedirs(self._result_save_dir, exist_ok=True)
            output_dir = os.path.join(self._result_save_dir, request_id)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'{request_id}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result_json_data, f, indent=2, ensure_ascii=False)
            self.get_logger().info(f'saved_result_json: {out_path}')
        except requests.RequestException as exc:
            self.get_logger().error(f'Failed to download result.json: {exc}')
        except (ValueError, OSError) as exc:
            self.get_logger().error(f'Failed to save result.json: {exc}')

    def _download_mp4(self, request_id: str):
        try:
            url = f'{self._model_api_url}/requests/{request_id}/wan_output.mp4'
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            output_dir = os.path.join(self._result_save_dir, request_id)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'{request_id}.mp4')
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            self.get_logger().info(f'saved_mp4: {out_path}')
        except requests.RequestException as exc:
            self.get_logger().error(f'Failed to download mp4: {exc}')
        except OSError as exc:
            self.get_logger().error(f'Failed to save mp4: {exc}')




def main():
    rclpy.init()
    node = ModelCommunicator()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
