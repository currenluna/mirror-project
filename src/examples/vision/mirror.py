#!/usr/bin/env python3


import argparse

from picamera import PiCamera

from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.annotator import Annotator


def avg_joy_score(faces):
    if faces:
        return sum(face.joy_score for face in faces) / len(faces)
    return 0.0

def main():
    """Face detection camera inference example."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', '-n', type=int, dest='num_frames', default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    args = parser.parse_args()

    # Forced sensor mode, 1640x1232, full FoV. See:
    # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
    # This is the resolution inference run on.
    
    # CHANGED CAMERA RESOLUTION
    with PiCamera(sensor_mode=5, resolution=(1640, 922), framerate=30) as camera:
        camera.hflip = True
        camera.start_preview()

        # Annotator renders in software so use a smaller size and scale results
        # for increased performace.
        annotator = Annotator(camera, bg_color='black', default_color='white', dimensions=(570, 320))
        scale_x = 570 / 1640
        scale_y = 320 / 922

        # Incoming boxes are of the form (x, y, width, height). Scale and
        # transform to the form (x1, y1, x2, y2).
        def transform(bounding_box):
            x, y, width, height = bounding_box
            return (scale_x * x, scale_y * y, scale_x * (x + width),
                    scale_y * (y + height))

        with CameraInference(face_detection.model()) as inference:
            for result in inference.run(args.num_frames):
                faces = face_detection.get_faces(result)
                annotator.clear()
                for face in faces:
                    annotator.bounding_box(transform(face.bounding_box), fill='black')
                annotator.update()
                #print('#%05d (%5.2f fps): num_faces=%d, avg_joy_score=%.2f' %
                #    (inference.count, inference.rate, len(faces), avg_joy_score(faces)))

        camera.stop_preview()


if __name__ == '__main__':
    main()
