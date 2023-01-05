from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
import torch
from PIL import Image
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config["SECRET_KEY"] = "asddfgh"
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"


transform = T.Compose([
    T.Resize(600),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

photos = UploadSet("photos", IMAGES)

configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, " Only images are allowed"),
            FileRequired("File Field should not be empty")

        ]
    )
    submit = SubmitField("Upload")


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(im, outputs, threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas_to_keep, bboxes_scaled


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



def plot_finetuned_results(pil_img, prob=None, boxes=None):
    finetuned_classes = ['balloon']
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    count_balloon = 0
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            cl = p.argmax()
            label = finetuned_classes[cl]
            if label == "balloon":
                count_balloon += 1
                text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False, color=c, linewidth=3))
                ax.text(xmin, ymin, text, fontsize=10,
                        bbox=dict(facecolor='yellow', alpha=0.5))
    print("Number of balloons detected: {}".format(count_balloon))
    plt.axis('off')
    plt.savefig('detr_results/new.jpg')
    return count_balloon
    


def run_worflow(my_image, my_model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    outputs = my_model(img)

    threshold = 0.75
    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(my_image, outputs, threshold=threshold)
    return plot_finetuned_results(my_image, probas_to_keep, bboxes_scaled)


def detr(filename):
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=1)
    checkpoint = torch.load('checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    im = Image.open(filename)
    return run_worflow(im, model)


def pretrained_model(filename):
    # Loading the model
    model = torch.hub.load('yolov5', 'custom', path='best.pt', force_reload=True, source='local')

    # Image processing
    img = filename

    # Model prediction
    results = model(img)
    results.save(save_dir='result/results')
    return results.pandas().xyxy[0].value_counts('name')




@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"], filename)


@app.route("/", methods=["GET", "POST"])
def home():
    form = UploadForm()
    file_url = None
    data = None

    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)
        # data = pretrained_model(f'uploads/{filename}')
        data = detr(f'uploads/{filename}')

    return render_template("index.html", form=form, file_url=file_url, data=data)


if __name__ == "__main__":
    app.run(debug=True)
