import numpy as np
from PIL import Image
from Model import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/images") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods= ["GET", "POST"])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        delete_path = "static/uploaded/"
        for file_name in os.listdir(delete_path):
            img_name = delete_path + file_name
            if os.path.isfile(img_name):
                os.remove(img_name)

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:8]  # Top 8 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        #print(scores)

        return render_template('index.html', query_path =uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")

# import numpy as np
# from PIL import Image
# from Model import FeatureExtractor
# from datetime import datetime
# from flask import Flask, jsonify, request, render_template
# from pathlib import Path

# import io
# import base64
# import os

# app = Flask(__name__)

# # Read image features
# fe = FeatureExtractor()
# features = []
# img_paths = []
# for feature_path in Path("./static/features").glob("*.npy"):
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./static/images") / (feature_path.stem + ".jpg"))
# features = np.array(features)

# def get_encoded_img(image_path):
#     with open(image_path, "rb") as img_file:
#         encoded_image = base64.b64encode(img_file.read())
    
#     return encoded_image

# @app.route('/', methods= ["GET", "POST"])
# def index():
#     if request.method == 'POST':
#         file = request.files['query_img']

#         # Save query image
#         img = Image.open(file.stream)  # PIL image
#         # uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
#         # img.save(uploaded_img_path)


#         # img = Image.open("./static/test/IMG_3535.jpg")

#         # Run search
#         query = fe.extract(img)
#         dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
#         ids = np.argsort(dists)[:3]  # Top 8 results
#         scores = [(dists[id], img_paths[id]) for id in ids]
#         #print(scores)

#         encoded_img_list = []
#         img_name_list = []
#         for score in scores:
#             encoded_img = get_encoded_img(score[1])
#             encoded_img_list.append(str(encoded_img))
#             img_name = os.path.basename(score[1])[:-4]
#             img_name_list.append(img_name)

#         return jsonify({"image_name": img_name_list, "encoded_img": encoded_img_list})
#     else:
#         return jsonify({'GET': True})


# if __name__=="__main__":
#     app.run("0.0.0.0")