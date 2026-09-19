import unittest
import sys
from unittest.mock import Mock, patch

import numpy as np
import cv2

from app import app, camera_lock


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_page_and_invalid_camera(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        with patch('app.cv2.VideoCapture') as capture:
            self.assertEqual(self.client.get('/stream?camera=-1').status_code, 400)
            capture.assert_not_called()

    def test_missing_camera_releases_lock(self):
        with patch('app.cv2.VideoCapture') as factory:
            capture = factory.return_value
            capture.isOpened.return_value = False
            self.assertEqual(self.client.get('/stream').status_code, 503)
            capture.release.assert_called_once()
            self.assertFalse(camera_lock.locked())

    def test_invalid_confidence_does_not_open_camera(self):
        with patch('app.cv2.VideoCapture') as factory:
            for value in ['nan', 'abc', '0.01', '1']:
                self.assertEqual(self.client.get(f'/stream?confidence={value}').status_code, 400)
            factory.assert_not_called()

    def test_detection_failure_releases_camera(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        with patch('app.cv2.VideoCapture', return_value=capture), patch('app.annotate', side_effect=RuntimeError('model missing')):
            with self.assertLogs('app', level='ERROR'):
                response = self.client.get('/stream?detect=1')
            self.assertEqual(response.status_code, 503)
            self.assertIn('model missing', response.json['error'])
            capture.release.assert_called_once()
            self.assertFalse(camera_lock.locked())

    def test_detection_annotates_each_frame(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, frame)
        with patch('app.cv2.VideoCapture', return_value=capture), patch('app.annotate', return_value=frame) as annotate:
            response = self.client.get('/stream?detect=1&confidence=0.6', buffered=False)
            try:
                iterator = iter(response.response)
                part = next(iterator)
                self.assertIn(b'X-Detection: on', part)
                self.assertIn(b'X-Detection-Result:', part)
                next(iterator)
                self.assertEqual(annotate.call_count, 2)
                self.assertEqual(annotate.call_args.args[1], 0.6)
            finally:
                response.close()
            self.assertFalse(camera_lock.locked())

    def test_stream_and_disconnect_release_camera(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
        with patch('app.cv2.VideoCapture', return_value=capture) as factory:
            response = self.client.get('/stream', buffered=False)
            try:
                self.assertEqual(response.status_code, 200)
                factory.assert_called_once_with(0, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
                self.assertIn(b'Content-Type: image/jpeg', next(response.response))
                self.assertEqual(self.client.get('/stream').status_code, 409)
            finally:
                response.close()
            capture.release.assert_called_once()
            self.assertFalse(camera_lock.locked())

    def test_demo_does_not_interrupt_live_camera(self):
        camera_lock.acquire()
        try:
            self.assertEqual(self.client.get('/detection-demo').status_code, 409)
        finally:
            camera_lock.release()


if __name__ == '__main__':
    unittest.main()
