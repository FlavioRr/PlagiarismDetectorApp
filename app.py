from flask import Flask, request, render_template
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import javalang

app = Flask(__name__)
MODEL_PATH = 'model/xgboost_model.pkl'
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = joblib.load(file)
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
    
    file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    file2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
    file1.save(file1_path)
    file2.save(file2_path)

    model = load_model()
    tfidf_vectorizer_path = 'preprocessing/preprocessed_all_data.pkl'
    _, _, tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

    def read_java_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

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
            print(f"Error parsing Java code: {e}")
            return ''

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
        result = "Plagiarism" if prediction[0] == 1 else "No Plagiarism"
        return render_template('result.html', result=result)
    else:
        return 'Error: AST extraction failed for one or both files.', 400

if __name__ == '__main__':
    app.run(debug=True)
