import cv2
from fer import FER, Video
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys

def main():
    face_detector = FER(mtcnn=True)
    input_video = Video(sys.argv[1])

    processing_data = input_video.analyze(face_detector, display=False)

    emo_data = input_video.to_pandas(processing_data)
    emo_data = input_video.get_first_face(emo_data)
    emo_data = input_video.get_emotions(emo_data)

    plt = emo_data.plot(figsize=(20, 8))

    fig = plt.get_figure()
    plt.ylabel('Emotion Intensity Value')
    plt.xlabel('Video Frame')
    plt.legend()
    plt.savefig(f'{sys.argv[1]}_plot.png')


if __name__ == '__main__':
    main()