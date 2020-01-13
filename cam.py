import time
import picamera
from datetime import datetime

class Cam:

    def snap(self):
        with picamera.PiCamera() as camera:
            camera.rotation = 180
            camera.resolution = (1024, 768)
            camera.start_preview()
            # Camera warm-up time
            time.sleep(2)
            camera.capture(self.generate_filename() + '.jpg')

    def generate_filename(self):
        return '/media/pi/footage/' + datetime.now().strftime("%Y%m%d_%H%M%S")
            