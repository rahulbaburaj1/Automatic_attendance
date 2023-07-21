import cv2
import face_recognition
import os
import numpy as np
import pickle
import csv
path = 'C:/Users/RAHUL/Desktop/Data/facerec/reco'

#def load_known_faces(known_faces_dir):
    
known_face_encodings = []
known_face_names = []

known_faces_dir = os.listdir(path)
print(known_faces_dir)
    
for file_name in known_faces_dir:
        #name, ext = os.path.splitext(file_name)
        #if ext == '.jpg':
           # image_path = os.path.join(known_faces_dir, file_name)
           # image = face_recognition.load_image_file(image_path)
           # face_encoding = face_recognition.face_encodings(image)[0]
           # known_face_encodings.append(face_encoding)
           # known_face_names.append(name)
    cur_img = cv2.imread(f'{path}/{file_name}')
    known_face_encodings.append(cur_img)
    name = os.path.splitext(file_name)[0]
    known_face_names.append(name)
print(len(known_face_encodings))
    
    #return known_face_encodings, known_face_names

def mark_attendance(name):
        with open("attendance.txt",'a') as file:
             file.write(name + "\n")
       

def find_encodings(known_face_encodings):
    
    encode_list = []
    for img in known_face_encodings:
          
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

available_faces = find_encodings(known_face_encodings)
        

#def recognize_faces(known_face_encodings, known_face_names):
video_capture = cv2.VideoCapture(0)

while True:
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(available_faces, face_encoding)
            distance = face_recognition.face_distance(available_faces, face_encoding)
            name = "Unknown"
            a = []

            if True in matches:
                #matched_idx = matches.index(True)
                matched_idx = np.argmin(distance)

                name = known_face_names[matched_idx]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            if name != "Unknown":
                #a.append([name])
                mark_attendance(name)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

for i in a:
     print(a[i])

#with open("attendance.csv",mode = 'w', newline='') as file:
        #writer = csv.writer(file)
        #writer.writerow(['Name'])
        #writer.writerows(a )
    # Provide the full or relative path to the "known_faces" directory here
    #known_faces_dir = os.path.join("C:/Users/RAHUL/Desktop/attendence/known_faces")

    #known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

   # if not known_face_encodings:
    #    print("No known faces found. Please add known faces to the 'known_faces' directory.")
   # else:
   #     recognize_faces(known_face_encodings, known_face_names)
