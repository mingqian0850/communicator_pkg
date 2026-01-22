import rclpy
from rclpy.node import Node

from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import base64
import json
import os
import requests


class ModelCommunicator(Node):
    def __init__(self):
        super().__init__('communicator_node')
        self.declare_parameter('image_topic', '/image_rect/compressed')
        self.declare_parameter('model_api_url', 'https://ea5i5e07fh1w-s8tvic3pz6qy.serving.hyperai.host')
        self.declare_parameter('prompt', 'Move to the monitor, stop in front of it, and avoid any collisions')
        self.declare_parameter('model_api_health_check', True)
        self.declare_parameter('model_api_timeout_sec', 600)
        self.declare_parameter('result_save_dir', '')
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
        if not self._result_save_dir:
            self._result_save_dir = os.path.join(self._get_workspace_root(), 'result')
        self._received_first = False
        self._done = False
        self._subscription = self.create_subscription(
            CompressedImage,
            self._image_topic,
            self._on_compressed_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f'Subscribed to image topic: {self._image_topic}')

    def _on_compressed_image(self, msg: CompressedImage):
        if self._received_first:
            return
        self._received_first = True
        self.get_logger().info('Received CompressedImage message, sending to server.')
        b64_str = base64.b64encode(msg.data).decode('utf-8')
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
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f'Invalid response from model API: {exc}')
        self.get_logger().info('Request finished, shutting down.')
        self.destroy_subscription(self._subscription)
        self._done = True

    @property
    def done(self):
        return self._done

    def _get_workspace_root(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(3):
            current_dir = os.path.dirname(current_dir)
        return current_dir

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
