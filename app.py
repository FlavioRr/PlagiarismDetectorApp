import os
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
MODEL_PATH = 'model/plagiarism_model.pkl'
UPLOAD_FOLDER = 'uploads'

# Crea el directorio de uploads si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'No files uploaded', 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    
    # Save uploaded files to 'uploads' directory
    file1.save(os.path.join(UPLOAD_FOLDER, file1.filename))
    file2.save(os.path.join(UPLOAD_FOLDER, file2.filename))

    # Perform plagiarism detection
    model = load_model()
    # Aquí iría tu lógica para comparar los archivos usando el modelo cargado.
    # Similarity calculation or feature extraction steps.
    result = "Simulated result"  # Reemplazar con el resultado real del modelo

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
