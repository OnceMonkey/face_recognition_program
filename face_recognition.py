'''
调用摄像头，实时人脸检测
    1. 加载已知人脸数据集库，转化成人脸向量和名称
    2. 通过face_encodings，将所有单张人脸图像转化为128的向量
    3. 使用时，将每张查找到的人脸向量与已知向量库进行对比，找出匹配且人脸距离最短的人脸
'''
import os
import cv2
import numpy as np
import face_recognition

from pathlib import Path

def load_known_db(known_face_dir):
    known_faces = list(Path(known_face_dir).rglob('*.*'))
    known_face_encodings = []
    known_face_names = []
    for path in known_faces:
        face_image = face_recognition.load_image_file(str(path))
        face_encoding = face_recognition.face_encodings(face_image)[0]
        face_name = path.parent.name
        known_face_encodings.append(face_encoding)
        known_face_names.append(face_name)
    return known_face_encodings,known_face_names

def real_time_recognition(known_face_dir):
    known_face_encodings, known_face_names  = load_known_db(known_face_dir)
    print('face database initialization completed!')
    print(f'total {len(known_face_names)} images, {len(set(known_face_names))} preson.')

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # locate all faces and get vectors.
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # in here, once detected faces, you can do other things.

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcamq
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    known_face_dir = r'./face_db'
    # load_known_db(known_face_dir)
    real_time_recognition(known_face_dir)
