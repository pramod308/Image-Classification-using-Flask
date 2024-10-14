import sys
import os
import codecs
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# Force UTF-8 encoding for output
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure Flask uses UTF-8

# Dictionary for class labels
dic = {0: 'cat', 1: 'dog', 2: 'horse', 3: 'human'}

# Load the saved Keras model
model = load_model('model.h5')
model.make_predict_function()

# Function to predict the label of an image
def predict_label(img_path):
    i = image.load_img(img_path, target_size=(100, 100))
    i = image.img_to_array(i)
    i = i.reshape(1, 100, 100, 3)
    i = i / 255.0
    p = model.predict(i)
    print(p)
    p = p.argmax()
    return dic[p]

# Home route
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
    return render_template("home.html")

# About route
@app.route("/about")
def about_page():
    return "About You..!!!"

# Submit route
@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']

        # Secure the filename to avoid encoding issues
        filename = secure_filename(img.filename.encode('utf-8').decode('utf-8'))
        print(filename)
        img_path = os.path.join('static/', filename)
        img.save(img_path)
        print(img_path)

        # Predict the label
        p = predict_label(img_path)

    return render_template("home.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
