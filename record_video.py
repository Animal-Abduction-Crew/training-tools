import picamera

with picamera.PiCamera() as camera:
    camera.rotation = 180
    camera.resolution = (1024, 768)
    camera.start_recording('my_video.h264')
    camera.wait_recording(5)
    camera.stop_recording()