import cv2
from fer import FER, Video
from fer import exceptions as fer_exceptions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import psutil
import time
import sys

def main():
    face_detector = FER(mtcnn=False)
    
    # record = {'timestamp': [], 'fps': [], 'cpu_percentage': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
    record = {'timestamp': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
    cpu_list = []
    fps_list = []

   
    #Get image seqence

    #Use camera feed if no file is specified
    capture = None
    if len(sys.argv) <= 1:
        capture = cv2.VideoCapture(0)
    else: 
        capture = cv2.VideoCapture(sys.argv[1])

    time_start = time.time()
    frameCount = 1
    while True:
        _, frame = capture.read()
        try:
            emotion, score = face_detector.top_emotion(frame)
        except fer_exceptions.InvalidImage:
            print("Invalid Image")
            break
        if emotion:
            frame = cv2.putText(frame, f"{emotion}: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("FER", frame)
            emotion_data = face_detector.detect_emotions(frame)[0]['emotions']
            frameCount += 1
            print(emotion_data)
            record['timestamp'].append(int(time.time() - time_start))
            cpu_list.append(psutil.cpu_percent(interval=1))
            # record['cpu_percentage'].append(psutil.cpu_percent(interval=1))

            fps_list.append(frameCount/int(time.time() - time_start))
            # record['fps'].append(frameCount/int(time.time() - time_start))

            for emo in emotion_data.keys():
                record[emo].append(emotion_data[emo])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


    # else:
    #     capture = cv2.VideoCapture(sys.argv[1])
    #     input_video = Video(sys.argv[1])
    #     processing_data = input_video.analyze(face_detector, display=True)



    #     emo_data = input_video.to_pandas(processing_data)
    #     emo_data = input_video.get_first_face(emo_data)
    #     emo_data = input_video.get_emotions(emo_data)

    #     ax = emo_data.plot(figsize=(20, 8))

    #     fig = ax.get_figure()
    #     ax.ylabel('Emotion Intensity Value')
    #     ax.xlabel('Video Frame')
    #     ax.legend()
    #     ax.savefig(f'{sys.argv[1]}_plot.png')

    df = pd.DataFrame.from_dict(record)
    df = pd.DataFrame.set_index(df, 'timestamp')
    print(df.head())

    fig, ax1 = plt.subplots()

    for column in df.columns[1:]:
        ax1.plot(df.index, df[column], label=column, zorder=2)

    ax1.set_xlabel('Time in seconds')
    ax1.set_ylabel('Emotion Score')

    ax2 = ax1.twinx()
    ax2.bar(df.index, cpu_list, alpha=0.5, label='CPU', zorder=1)
    ax2.set_ylabel('CPU Usage in Percentage')

    textstr = f'FPS: {round(fps_list[0], 3)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Place the text box at position x=0.05, y=0.95, relative to the figure size
    ax1.text(0.5, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.title('Detected Emotion')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()