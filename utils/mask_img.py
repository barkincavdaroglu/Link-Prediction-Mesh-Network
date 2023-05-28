import numpy as np
from math import cos, sin, radians, degrees, atan2, sqrt, pi, log, asin
import cv2
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class MaskImg(object):
    def __init__(self, edge_index, p1, p2, w, h):
        self.p1 = p1
        self.p2 = p2

        self.center = self.center_geolocation([p1, p2])
        self.zoom = self.getZoomLevel()

        self.w = w
        self.h = h
        self.edge_index = edge_index

    def get_map_image(self):
        api_key = os.getenv("GOOGLE_MAP_API")

        url = "https://maps.googleapis.com/maps/api/staticmap?"

        center = str(self.center[0]) + "%2C" + str(self.center[1])

        visible_points = (
            str(self.p1[0])
            + "%2C"
            + str(self.p1[1])
            + "%7C"
            + str(self.p2[0])
            + "%2C"
            + str(self.p2[1])
        )

        r = requests.get(
            url
            + "center="
            + center
            + "&visible="
            + visible_points
            + "&zoom="
            + str(self.zoom)
            + "&size="
            + str(self.w)
            + "x"
            + str(self.h)
            + "&maptype=satellite"
            + "&key="
            + api_key
        )

        # wb mode is stand for write binary mode
        f = open(
            "/Users/barkincavdaroglu/Desktop/Link-Prediction-Mesh-Network/edge_imgs/"
            + str(self.edge_index)
            + ".png",
            "wb",
        )
        f.write(r.content)
        f.close()

        return (
            "/Users/barkincavdaroglu/Desktop/Link-Prediction-Mesh-Network/edge_imgs/"
            + str(self.edge_index)
            + ".png"
        )

    def center_geolocation(self, geolocations):
        x = 0
        y = 0
        z = 0

        for lat, lon in geolocations:
            lat = radians(float(lat))
            lon = radians(float(lon))
            x += cos(lat) * cos(lon)
            y += cos(lat) * sin(lon)
            z += sin(lat)

        x = float(x / len(geolocations))
        y = float(y / len(geolocations))
        z = float(z / len(geolocations))

        return (degrees(atan2(z, sqrt(x * x + y * y))), degrees(atan2(y, x)))

    def get_pixel(self, lat, long):
        """
        x, y - location in degrees
        x_center, y_center - center of the map in degrees (same value as in the google static maps URL)
        zoom_level - same value as in the google static maps URL
        x_ret, y_ret - position of x, y in pixels relative to the center of the bitmap
        """
        OFFSET = (
            268435456  # half of the earth circumference's in pixels at zoom level 21
        )
        RADIUS = OFFSET / pi

        def l_to_x(x):
            return int(round(OFFSET + RADIUS * x * pi / 180))

        def l_to_y(y):
            return int(
                round(
                    OFFSET
                    - RADIUS
                    * log((1 + sin(y * pi / 180)) / (1 - sin(y * pi / 180)))
                    / 2
                )
            )

        x_ret = (l_to_x(long) - l_to_x(self.center[0])) >> (21 - self.zoom)
        y_ret = (l_to_y(lat) - l_to_y(self.center[1])) >> (21 - self.zoom)
        return x_ret + self.w, y_ret + self.h

    @staticmethod
    def is_inside(x, y, theta, h, k, a, b):
        return ((cos(theta) * (x - h) + sin(theta) * (y - k)) ** 2) / (a**2) + (
            (sin(theta) * (x - h) - cos(theta) * (y - k)) ** 2
        ) / (b**2) <= 1

    @staticmethod
    def haversine_dist(p1, p2):
        # The math module contains a function named
        # radians which converts from degrees to radians.
        lon1 = radians(p1[0])
        lon2 = radians(p2[0])
        lat1 = radians(p1[1])
        lat2 = radians(p2[1])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371

        # calculate the result
        return c * r

    @staticmethod
    def distance(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def getZoomLevel(self):
        latdiff = abs(self.p1[0] - self.p2[0])
        londiff = abs(self.p1[1] - self.p2[1])

        maxdiff = max(latdiff, londiff)
        if maxdiff < 360 / 2**20:
            return 21
        else:
            zoom = round(-1 * ((log(maxdiff) / log(2)) - (log(360) / log(2))))
            if zoom < 1:
                zoom = 1
            return zoom

    def maskedImage(self):
        loc = self.get_map_image()

        # Load image, create mask, and draw white circle on mask
        image = cv2.imread(loc)

        mask = [
            [[0, 0, 0] for i in range(image.shape[1])] for j in range(image.shape[0])
        ]

        p1_pix, p2_pix = self.get_pixel(self.p1[0], self.p1[1]), self.get_pixel(
            self.p2[0], self.p2[1]
        )
        a = self.distance(p1_pix, p2_pix) / 2

        b = 8.656 * sqrt((a) / 0.915)
        print("a: {a}, b: {b}".format(a=a, b=b))

        h, k = (self.w / 2, self.h / 2)

        angle = asin(abs(p2_pix[1] - p1_pix[1]) / self.distance(p1_pix, p2_pix))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if not self.is_inside(i, j, angle, h, k, a / 2, a):
                    mask[i][j] = [0, 0, 0]
                else:
                    mask[i][j] = [255, 255, 255]

        mask = np.array(mask)
        mask = mask.astype(np.uint8)

        # Mask input image with binary mask
        result = cv2.bitwise_and(image, mask)

        # Save result
        cv2.imwrite(
            "/Users/barkincavdaroglu/Desktop/Link-Prediction-Mesh-Network/masked_edge_imgs/"
            + str(self.edge_index)
            + ".png",
            result,
        )


obj = MaskImg(1, (43.704867, -72.290485), (43.705371, -72.289514), 400, 400)
obj.maskedImage()
