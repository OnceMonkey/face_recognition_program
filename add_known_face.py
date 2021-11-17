'''
将视频中的人脸进行录入到已知人脸数据库中
以视频的方式录制人脸：
    1.用户输入用户名，建立目录
    2.打开摄像头，定位摄像头中的第一张人脸
    3.如果用户按下s，就保存人脸

以图片集的方式录入人脸：
    直接将每个目录复制到已知人脸数据库中就可以了。
'''

import os
import cv2
import numpy as np
import face_recognition

import time

def record_from_video(save_dir):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        fram_show = cv2.copyTo(frame,None)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations)>0:
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(fram_show, (left, top), (right, bottom), (0, 0, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                save_path = os.path.join(save_dir,f'{int(time.time())}.png')
                # 保存原图
                cv2.imwrite(save_path,frame)
                # 保存指定大小图, 以加快加载速度
                cv2.imwrite(save_path,frame)
                print(f'save successfully! -> {save_path}')

        cv2.imshow('Video', fram_show)
        cv2.waitKey(1)

    # Release handle to the webcamq
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    save_dir=r'./face_db/obama'
    if not os.path.exists(save_dir):os.mkdir(save_dir)

    record_from_video(save_dir)