import cv2
import time
import requests
import base64

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return None

    # Crop the first face found
    (x, y, w, h) = faces[0]
    x = x - 10
    y = y - 10
    cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0
name = input("Enter Name:")
# GitHub repository information
repo_owner = "rishabhRsinghvi"
repo_name = "SmartAttendance"
file_path = "Images"

# Collect 10 samples of the face from the webcam input
while True:
    ret, frame = cap.read()
    face_result = face_extractor(frame)

    if face_result is None:
        print("Face not found")
        pass

    else:
        count += 1
        face = cv2.resize(face_result, (400, 400))
        # Put count on images
        cv2.putText(face, str(count), (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

        # Convert image to base64
        _, img_encoded = cv2.imencode('.jpg', face)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Upload image to GitHub
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}/{name}_{count}.jpg"
        headers = {'Authorization': 'token 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'}  # Replace with your GitHub token
        data = {
            'message': f'Add image {count} for {name}',
            'content': img_base64
        }
        response = requests.put(url, headers=headers, json=data)

        time.sleep(1)

    if cv2.waitKey(1) == 13 or count == 10:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Completed Collecting Samples!!!")
