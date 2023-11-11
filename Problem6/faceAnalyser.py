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
    
    record = {'timestamp': [], 'angry': [], 'disgust': [], 'fear': [], 'happy': [], 'sad': [], 'surprise': [], 'neutral': []}
    cpu_list = []
    fps_list = []

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
            if len(sys.argv) <= 1:
                frame = cv2.putText(frame, f"{emotion}: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("FER", frame)
            emotion_data = face_detector.detect_emotions(frame)[0]['emotions']
            frameCount += 1
            # print(emotion_data)
            record['timestamp'].append(int(time.time() - time_start))
            cpu_list.append(psutil.cpu_percent())
            # record['cpu_percentage'].append(psutil.cpu_percent(interval=1))
            try:
                fps_list.append(frameCount/int(time.time() - time_start))

            except:
                print('failed to append to fps_list')
            # record['fps'].append(frameCount/int(time.time() - time_start))

            for emo in emotion_data.keys():
                record[emo].append(emotion_data[emo])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

    df = pd.DataFrame.from_dict(record)
    df = pd.DataFrame.set_index(df, 'timestamp')
    print(df.head())

    fig, ax1 = plt.subplots()

    for column in df.columns:
        ax1.plot(df.index, df[column], label=column, zorder=2)

    ax1.set_xlabel('Time in seconds')
    ax1.set_ylabel('Emotion Score')

    # Get rolling average of CPU
    smoothed_CPU = []
    window = 20
    for i in range(len(cpu_list) - window + 1):
        # Calculate the average of current window
        window_average = sum(cpu_list[i:i+window]) / window
        # Append to the list of moving averages
        smoothed_CPU.append(window_average)


    ax2 = ax1.twinx()
    ax2.plot(df.index[window-1:], smoothed_CPU, '-k', alpha=1, label='CPU', zorder=1)
    ax2.set_ylabel('CPU Usage in Percentage')

    avg_fps = sum(fps_list) / len(fps_list)
    textstr = f'FPS: {round(avg_fps, 3)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Place the text box at position x=0.05, y=0.95, relative to the figure size
    ax1.text(0.5, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.title('Detected Emotion')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)
    if len(sys.argv) <= 1:
        plt.savefig(f"./Results/{str(time.time())}.png")
    else: 
        plt.savefig("./Results/latest.png")

if __name__ == '__main__':
    main()