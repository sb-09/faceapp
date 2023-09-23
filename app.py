import cv2
import dlib
import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import face_recognition

app = Flask(__name__)

UPLOAD_FOLDER = 'face/static/uploads'
PROCESSED_FOLDER = 'face/static/processed'

app.config['face/static/uploads'] = UPLOAD_FOLDER
app.config['face/static/processed'] = PROCESSED_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'photo' not in request.files:
        return redirect(url_for('index'))

    photo = request.files['photo']

    if photo.filename == '':
        return redirect(url_for('index'))

    # Save uploaded photo
    photo_path = os.path.join(app.config['face/static/uploads'], 'photo.jpg')
    photo.save(photo_path)

    # Perform image processing (e.g., resizing, filters)
    processed_image = process_image(photo_path)

    # Save processed photo
    processed_path = os.path.join(app.config['face/static/processed'], 'processed.jpg')
    processed_image.save(processed_path)

    return render_template('identify.html', processed_path=processed_path)

@app.route('/identify')
def identify():
    # Load the processed image
    processed_path = os.path.join(app.config['face/static/processed'], 'processed.jpg')
    processed_image = Image.open(processed_path)

    # Identify faces in the processed image
    face_locations = face_recognition.face_locations(processed_image)
    num_faces = len(face_locations)

    return render_template('identify.html', processed_path=processed_path, num_faces=num_faces)

@app.route('/extract', methods=['POST'])
def extract_faces():
    # Load the processed image
    processed_path = os.path.join(app.config['face/static/processed'], 'processed.jpg')
    processed_image = face_recognition.load_image_file(processed_path)

    # Find face locations in the processed image
    face_locations = face_recognition.face_locations(processed_image)

    # Extract faces
    extracted_faces = []
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        face_image = processed_image[top:bottom, left:right]
        face_image = Image.fromarray(face_image)
        extracted_faces.append(face_image)
        face_image.save(os.path.join(app.config['face/static/uploads'], f'face_{i}.jpg'))

    return render_template('extract.html', processed_path=processed_path)

@app.route('/upload2', methods=['POST'])
def upload_second_photo():
    if 'second_photo' not in request.files:
        return redirect(url_for('index'))

    second_photo = request.files['second_photo']

    if second_photo.filename == '':
        return redirect(url_for('index'))

    # Save the second uploaded photo
    second_photo_path = os.path.join(app.config['face/static/uploads'], 'second_photo.jpg')
    second_photo.save(second_photo_path)

    return render_template('swap.html', processed_path=processed_path, second_photo_path=second_photo_path)

@app.route('/swap_faces', methods=['POST'])
def swap_faces():
    # Load the processed image and second photo
    processed_path = os.path.join(app.config['face/static/processed'], 'processed.jpg')
    processed_image = cv2.imread(processed_path)
    second_photo_path = os.path.join(app.config['face/static/uploads'], 'second_photo.jpg')
    second_photo = cv2.imread(second_photo_path)

    # Perform face detection with dlib on the processed image
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    faces = detector(processed_gray)

    # Ensure exactly one face is detected in the processed image
    if len(faces) != 1:
        return "Exactly one face must be detected in the processed image."

    # Extract facial landmarks for the face in the processed image
    landmarks = predictor(processed_gray, faces[0])

    # Perform face detection with OpenCV on the second photo
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    second_photo_gray = cv2.cvtColor(second_photo, cv2.COLOR_BGR2GRAY)
    faces_cv2 = face_cascade.detectMultiScale(second_photo_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Ensure exactly one face is detected in the second photo
    if len(faces_cv2) != 1:
        return "Exactly one face must be detected in the second photo."

    # Extract the ROI (Region of Interest) from the second photo containing the detected face
    x, y, w, h = faces_cv2[0]
    face_roi = second_photo[y:y+h, x:x+w]

    # Resize the extracted face ROI to match the size of the face in the processed image
    resized_face_roi = cv2.resize(face_roi, (faces[0].width(), faces[0].height()))

    # Swap the faces
    swapped_image = processed_image.copy()
    swapped_image[y:y+h, x:x+w] = resized_face_roi

    # Save the swapped result
    result_path = os.path.join(app.config['face/static/processed'], 'result.jpg')
    cv2.imwrite(result_path, swapped_image)

    return render_template('result.html', result_path=result_path)



def process_image(image_path):
    # Load the image using PIL (Pillow)
    image = Image.open(image_path)

    # Implement image processing (e.g., resizing, filters) here
    # Example: resized_image = image.resize((width, height))

    return image

if __name__ == '__main__':
    app.run(debug=True)
