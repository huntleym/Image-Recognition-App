from flask import Flask, jsonify, request
from app.torch_utils import transform_image, get_prediction
import traceback

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
	# xxx.png
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		file = request.files.get('file')
		if file is None or file.filename == "":
			return jsonify({'error': 'no file'})
		if not allowed_file(file.filename):
			return jsonify({'error': 'format not supported'})

		try:
			img_bytes = file.read()
			tensor = transform_image(img_bytes) #problem at tranforming image
			prediction = get_prediction(tensor)
			data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
			return jsonify(data)
		except:
			traceback.print_exc()
			return jsonify({'error': 'error during prediction'})

@app.route('/hello')
def hello():
	return 'Hello world!'
#
# if __name__ == '__main__':
# 	app.run()
