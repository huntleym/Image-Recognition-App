import os
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, flash, request, redirect, send_file, render_template
from torch_utils import transform_image, get_prediction #app.torch_utils
import traceback

UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__, template_folder='templates')
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/uploadfile', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file found."
		file = request.files['file']
		# if user does not select file, browser submits empty part without filename
		if file.filename == '':
			return "No file name found."
		if not allowed_file(file.filename):
			return "File type not allowed."

		try:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			print('saved file successfully')
			# send file name as parameter to download
			#return redirect('/downloadfile/' + filename)
			#return redirect('/displayresults/', value = filename)
			return "Saved file successfully!"
		except:
			traceback.print_exc()
			return jsonify({'error': 'saving file'})

	return render_template('index.html')

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

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = UPLOAD_FOLDER + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')

@app.route('/')
def index():
	#return "Flask app deployed"
    return render_template('index.html')

@app.route('/hello')
def hello():
	return 'Hello world!'

if __name__ == "__main__":
    app.run(debug = True)
