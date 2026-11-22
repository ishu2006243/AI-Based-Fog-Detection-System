import json
from django.test import TestCase, Client
from django.conf import settings


class FogDetectorTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_homepage_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Glass Fog Detector")

    def test_api_detect_sample_image_fogged(self):
        payload = json.dumps({"sample_name": "image.png"})
        response = self.client.post("/api/detect/", data=payload, content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "fogged")
        self.assertTrue(data["is_fogged"])
        self.assertIn("serial", data)
        self.assertIn("sent", data["serial"])
        self.assertIn("message", data["serial"])

    def test_api_detect_sample_image3_clear(self):
        payload = json.dumps({"sample_name": "image3.png"})
        response = self.client.post("/api/detect/", data=payload, content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "clear")
        self.assertFalse(data["is_fogged"])
        self.assertNotIn("serial", data)

    def test_api_detect_sample_image4_clear(self):
        payload = json.dumps({"sample_name": "image4.png"})
        response = self.client.post("/api/detect/", data=payload, content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "clear")
        self.assertFalse(data["is_fogged"])

    def test_api_upload_file(self):
        image_path = settings.BASE_DIR / "image4.png"
        with open(image_path, "rb") as f:
            response = self.client.post("/api/detect/", {"image": f})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "clear")

    def test_serial_status_endpoint(self):
        response = self.client.get("/api/serial/status/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["pyserial_available"])
        self.assertIn("available_ports", data)

    def test_serial_mock_send_command(self):
        from unittest.mock import MagicMock, patch
        from detector.serial_controller import SerialController

        controller = SerialController.get_instance()
        mock_serial_conn = MagicMock()
        mock_serial_conn.is_open = True

        with patch("serial.Serial", return_value=mock_serial_conn):
            with patch("detector.serial_controller.detect_candidate_port", return_value="/dev/ttyUSB0"):
                controller.close()
                ok, msg = controller.send_command("*1")
                self.assertTrue(ok)
                mock_serial_conn.write.assert_called_once_with(b"*1")
                mock_serial_conn.flush.assert_called_once()
                controller.close()

