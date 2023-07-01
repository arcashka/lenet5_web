from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import base64
import io
from PIL import Image

import lenet5

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'no image'}), 400

    file = request.files['image']

    image_path = 'images/received_image.jpg'
    file.save(image_path)

    image = convert_image(image_path)
    result = process_received_image(image)

    return jsonify({
        'result': result,
        'processed_image': numpy_to_base64(image)
    })

def numpy_to_base64(np_array):
    np_array = np_array[0, 0, :, :]
    img = Image.fromarray(np_array.astype('uint8'))
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_bytes = img_io.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_b64

def convert_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = image.convert("L")
    image = np.array(image)
    return image.reshape((1, 1, 32, 32))

def process_received_image(image):
    model = lenet5.LeNet5(10).to('cuda')
    model.load_state_dict(torch.load('models/20_epochs'))
    model.eval()

    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.to('cuda')
    prediction = model(image_tensor)
    labels = torch.argmax(prediction[0], 1)
    return str(labels[0].item())

if __name__ == '__main__':
    ssh_folder = "/home/arcashka/.ssh/self_signed_certs/"
    app.run(host='0.0.0.0', port=5000, ssl_context=(ssh_folder + 'cert.pem', ssh_folder + 'key.pem'))

