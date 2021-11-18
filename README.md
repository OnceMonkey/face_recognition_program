# face_recognition_program
 This is a simple face recognition program based on "ageitgey/face_recognition", thanks for their great work. Most of codes are heavily same, this repository just for run right out of the box.


# usage
## 1. Add known faces
All known face images have been placed at dir "face_db".
Add some persons's face into it, like:

```
--face_db
----person1
------0.png
------1.png
----person2
----person3
```

## 2. run face_recognition.py

`python face_recognition.py`

The result looks similar to the following：
<img src="https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif" alt="show" />

# reference
- https://github.com/ageitgey/face_recognition
- https://github.com/ageitgey/face_recognition_models
- [https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
