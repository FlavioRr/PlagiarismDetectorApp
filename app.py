from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import numpy as np
import javalang
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'xgboost_model.pkl')
TFIDF_VECTOR_PATH = os.path.join(os.path.dirname(__file__), 'preprocessing', 'preprocessed_all_data.pkl')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)
    return model

def load_vectorizer():
    if not os.path.exists(TFIDF_VECTOR_PATH):
        raise FileNotFoundError(f"Vectorizer file not found: {TFIDF_VECTOR_PATH}")
    with open(TFIDF_VECTOR_PATH, 'rb') as file:
        data = joblib.load(file)
        tfidf_vectorizer = data[2]  # Suponiendo que el vectorizador es el tercer elemento
    return tfidf_vectorizer

def read_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def extract_ast_nodes(code):
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse()

        ast_nodes = []
        for path, node in tree:
            if isinstance(node, javalang.tree.Node):
                ast_nodes.append(node.__class__.__name__)
        return ' '.join(ast_nodes)
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
        print(f"Error al analizar el código Java: {e}")
        return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files or 'file2' not in request.files:
        return 'No se subieron archivos', 400

    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if not file1.filename.endswith('.java') or not file2.filename.endswith('.java'):
        return 'Por favor, sube solo archivos .java', 400

    file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    file2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
    file1.save(file1_path)
    file2.save(file2_path)

    try:
        model = load_model()
        tfidf_vectorizer = load_vectorizer()
    except FileNotFoundError as e:
        return str(e), 500

    code1 = read_java_file(file1_path)
    code2 = read_java_file(file2_path)
    ast_nodes1 = extract_ast_nodes(code1)
    ast_nodes2 = extract_ast_nodes(code2)

    if ast_nodes1 and ast_nodes2:
        combined_ast_nodes = ast_nodes1 + ' ' + ast_nodes2
        combined_features = tfidf_vectorizer.transform([combined_ast_nodes])
        
        ast_set1 = set(ast_nodes1.split())
        ast_set2 = set(ast_nodes2.split())
        
        jaccard_similarity = len(ast_set1.intersection(ast_set2)) / len(ast_set1.union(ast_set2)) if len(ast_set1.union(ast_set2)) != 0 else 0
        vec1 = tfidf_vectorizer.transform([ast_nodes1]).toarray()[0]
        vec2 = tfidf_vectorizer.transform([ast_nodes2]).toarray()[0]
        manhattan_distance = np.sum(np.abs(vec1 - vec2))
        
        additional_features = np.array([[manhattan_distance, jaccard_similarity]])
        combined_features = np.hstack((combined_features.toarray(), additional_features))

        prediction = model.predict(combined_features)
        result = "Plagio" if prediction[0] == 1 else "No Plagio"
        return render_template('result.html', result=result)
    else:
        return 'Error: la extracción del AST falló para uno o ambos archivos.', 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
