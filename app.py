from flask import Flask,render_template, send_from_directory,url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed,FileRequired
from wtforms import SubmitField
import torch

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

def pretrained_model(filename):

    # Loading the model
    model = torch.hub.load('yolov5','custom', path='best.pt',force_reload=True,source='local')

    # Image processing
    # img = "uploads/audrey-martin-FJpHcqMud_Y-unsplash.jpg"
    img = filename

    # Model prediction
    results = model(img)
    results.save(save_dir='results')    
    print(results.pandas().xyxy[0].value_counts('name'))
    


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"],filename) 



@app.route("/",methods=["GET","POST"])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file",filename=filename)
        pretrained_model(f'uploads/{filename}')
    else:
        file_url=None
    return render_template("index.html",form=form,file_url=file_url)


       

if __name__ == "__main__":
    app.run(debug=True)