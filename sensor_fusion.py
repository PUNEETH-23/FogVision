"""
sensor_fusion.py
----------------
Camera-ESP32 Sensor Fusion (Plan A).

Combines camera detections with physical VL53L0X distance measurements
obtained from the ESP32 server endpoints.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import esp32_module


class SensorFusion:
    """
    Fuses camera object detections with physical ESP32 zone-based distance sensors.
    """

    def __init__(self):
        self.sensor_connected = False
        # Conflict threshold parameter (meters)
        self.sensor_camera_tolerance = 5.0

    def fuse_detections(self, detections: List[Dict[str, Any]],
                        is_live: bool = False,
                        esp32_ip: str = "") -> Dict[str, Any]:
        """
        Fuse camera detections with physical ESP32 distance sensors.

        Parameters
        ----------
        detections : list
            YOLO detections from object_detect.py
        is_live : bool
            True if running live camera feed mode
        esp32_ip : str
            IP address of the ESP32 server

        Returns
        -------
        Dict with fused detections, ESP32 sensor values, and statistics.
        """
        # Initialize default sensor values (80.0 m = "no reading" sentinel)
        esp32_sensors = {"left": 80.0, "middle": 80.0, "right": 80.0}

        # Query physical ESP32 sensors if active live feed
        self.sensor_connected = False
        if is_live and esp32_ip:
            real_data = esp32_module.get_sensor_data(esp32_ip, timeout=0.5)
            if real_data is not None:
                esp32_sensors["left"]   = real_data["left"]
                esp32_sensors["middle"] = real_data["middle"]
                esp32_sensors["right"]  = real_data["right"]
                self.sensor_connected = True

        esp32_sensors = {k: round(v, 2) for k, v in esp32_sensors.items()}

        # Fuse distances with detections based on horizontal sections
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) // 2

            if cx < 213:
                obj_section = "left"
            elif cx < 426:
                obj_section = "middle"
            else:
                obj_section = "right"

            sensor_dist = esp32_sensors[obj_section]
            d["distance_sensor"] = sensor_dist
            d["sensor_section"]  = obj_section

            # In live mode use ESP32 sensor as the primary distance source
            if is_live and self.sensor_connected:
                d["distance"]        = sensor_dist
                d["distance_bbox"]   = sensor_dist
                d["distance_source"] = "esp32_sensor"
            else:
                if "distance_source" not in d:
                    d["distance_source"] = "bounding_box"

        # Calculate statistics
        fusion_stats = {
            "total_detections":      len(detections),
            "esp32_distances_used":  len(detections) if (is_live and self.sensor_connected) else 0,
            "vision_distances_used": 0 if (is_live and self.sensor_connected) else len(detections),
            "esp32_connected":       self.sensor_connected
        }

        return {
            "fused_detections": detections,
            "esp32_sensors":    esp32_sensors,
            "conflicts":        [],
            "fusion_stats":     fusion_stats
        }

    def get_road_context_fusion(self) -> Dict[str, Any]:
        """
        Retrieves road context data.
        """
        return {
            "road":             "Unknown Road",
            "road_type":        "unknown",
            "curvature":        0.0,
            "curvature_radius": 9999.0,
            "sensor_source":    "esp32",
            "confidence":       0.5,
            "esp32_connected":  self.sensor_connected,
            "blackspots":       []
        }

    def get_fusion_health(self) -> Dict[str, Any]:
        """
        Get health status of sensor fusion system.
        """
        return {
            "esp32_status":      "connected" if self.sensor_connected else "not_connected",
            "camera_status":     "operational",
            "fusion_confidence": 0.8 if self.sensor_connected else 0.4
        }