from flask import Flask,render_template, send_from_directory,url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed,FileRequired
from wtforms import SubmitField
import torch
import os
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image
import face_recognition
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "asddfgh"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

photos = UploadSet("photos", IMAGES)
configure_uploads(app,photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators = [
            FileAllowed(photos, " Only images are allowed"),
            FileRequired("File Field should not be empty")
        ] 
    )
    submit = SubmitField("Upload")


def yolo(filename):
    # Loading the model
    model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', force_reload=True, source='local')
    # Image processing
    img = f'uploads/{filename}'
    # Model prediction
    results = model(img)  
    # Save result
    results.save(save_dir = f'result/balloon_processing/{filename}')
    try:
        number =  results.pandas().xyxy[0].value_counts('name').tolist()[0] 
    except:
        number = 0
    finally:
        if number >= 5:
            if number >= 10:
                output = f'Wow, you have so many balloons.'
            else:
                output = f'Wow, you have {number} balloons.'
        elif number == 0:
            output = f"Oh, I don't see any balloon."
        elif number == 1:
            output = f'I see {number} balloon. We could definitely use more balloons!'
        else:
            output = f'I see {number} balloons. We could definitely use more balloons!'
        return output, number

def happyface_model(filename, NumberofBalloons):
    # load json and create model
    json_file = open('dcnn-happy-face/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_face = model_from_json(loaded_model_json)
    # load weights into new model
    model_face.load_weights("dcnn-happy-face/model_weights.h5")
    # Image processing - crop face, turn into greyscale
    # Get face array
    image_array = face_recognition.load_image_file(f'uploads/{filename}')
    face_locations = face_recognition.face_locations(image_array)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        coordinates = image_array[top:bottom, left:right]
        face = Image.fromarray(coordinates)
        # save the face as processing result
        try:
            os.mkdir('result/face_processing')
        except:
            pass
        face.save(f'result/face_processing/{filename}_face.jpg')
    # Convert into greyscale & reshape array
    try:
        face_grey = Image.open(f'result/face_processing/{filename}_face.jpg').convert('L')
        face_grey.save(f'result/face_processing/{filename}_greyscale.jpg')
        # Image processing - resize, normalize, reshape
        img = cv2.imread(f'result/face_processing/{filename}_greyscale.jpg')
        img_resized = cv2.resize(img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
        face_greyscale_array_new = np.delete(img_resized, (1,2), 2)
        # Normalize the data, because in the model we normalized our faces
        face_greyscale_array_normalized = face_greyscale_array_new / 255.
        face_greyscale_array_normalized.shape
        # Reshape
        face_greyscale_array_normalized_reshaped = face_greyscale_array_normalized.reshape(1,48,48,1)
        img_2_predict = face_greyscale_array_normalized_reshaped
        # Model prediction
        mapper = {
            0: "Not Happy",
            1: "Happy"
        }
        results = mapper[np.argmax(model_face.predict(img_2_predict), axis=-1)[0]]
        # Return prediction
        if NumberofBalloons == 0:
            if results=='Happy':
                output = 'You seem happy. What is going on ^_^'
            else:
                output = "I can feel you're not in a cheerful mood -_-' should I tell you a joke?"
        else:
            if results=='Happy':
                output = 'Great party! Let me turn on the Party Playlist ^_^'
            else:
                output = "Oh you look not super in the mood for celebration. How can I help ~_~?"
        return output
    except:
        if NumberofBalloons == 0:
            output = ''
        else:
            output = 'Attention please! I also want to join the party.'
        return output

@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"],filename) 

@app.route("/", methods=["GET", "POST"])
def home():
    form = UploadForm()
    file_url = None
    data = None
    data_face = None
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file",filename=filename)
        data, number = yolo(filename)
        data_face = happyface_model(filename, number)
    return render_template("index.html",form=form,file_url=file_url,data=data,data_face=data_face)

if __name__ == "__main__":
    app.run(debug=True)