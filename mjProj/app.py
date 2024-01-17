from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50

app = Flask(__name__)

model = VGG16()
#model= ResNet50()
@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/pred', methods=['POST'])
def predict():
    
    imageFile = request.files['imagefile']
    imagePath = './static/' + imageFile.filename
    imageFile.save(imagePath)

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    # print(imageFile.filename)

    return render_template('index.html', prediction=classification, img_name=imageFile.filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)