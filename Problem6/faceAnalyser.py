import cv2
from fer import FER, Video
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import psutil
import time
import sys

def main():
    face_detector = FER(mtcnn=True)
    
    record = {'timestamp': [], 'cpu_percentage': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
    # Record format: (frame, timestamp, emotionData, cpuUsage)

    #Use camera feed if no file is specified
    if len(sys.argv) <= 1:
        capture = cv2.VideoCapture(0)
        time_start = time.time()
        frameCount = 1
        while True:
            _, frame = capture.read()
            emotion, score = face_detector.top_emotion(frame)
            if emotion:
                frame = cv2.putText(frame, f"{emotion}: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("FER", frame)
                emotion_data = face_detector.detect_emotions(frame)[0]['emotions']
                frameCount += 1
                cpu_usage = psutil.cpu_percent(interval=1)
                print(emotion_data)
                record['timestamp'].append(time.time() - time_start)
                record['cpu_percentage'].append(psutil.cpu_percent())
                for emo in emotion_data.keys():
                    record[emo].append(emotion_data[emo])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()

        #Process and plot data
        # print(f"record: \n{record}")
        df = pd.DataFrame.from_dict(record)
        print(df.head())
        plt = df.plot(x='timestamp', y=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        plt.title('emotions')
        plt.show()
    else:
        capture = cv2.VideoCapture(sys.argv[1])
        input_video = Video(sys.argv[1])
        processing_data = input_video.analyze(face_detector, display=True)

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