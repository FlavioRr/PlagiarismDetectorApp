import os
import pandas as pd
import javalang
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Function to read a Java file and return its content as a string
def read_java_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Function to extract AST nodes from Java code
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

# Function to calculate Jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to calculate Manhattan distance between two vectors
def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

# Function to preprocess all Java files in the specified directory and return a DataFrame
def preprocess_data_with_ast(java_files_dir, labels_file):
    data = []
    labels = pd.read_csv(labels_file)
    
    print("Available columns in labels.csv:", labels.columns)
    
    label_column = 'veredict'  # Ensure the correct column is present
    if label_column not in labels.columns:
        raise KeyError(f"The column '{label_column}' is not found in labels.csv")
    
    for index, row in labels.iterrows():
        sub1 = row['sub1']
        sub2 = row['sub2']
        label = row[label_column]  # Use the correct label column
        
        # Paths to the two Java files
        file1_path = os.path.join(java_files_dir, f"{sub1}.java")
        file2_path = os.path.join(java_files_dir, f"{sub2}.java")
        
        if os.path.exists(file1_path) and os.path.exists(file2_path):
            code1 = read_java_file(file1_path)
            code2 = read_java_file(file2_path)
            ast_nodes1 = extract_ast_nodes(code1)
            ast_nodes2 = extract_ast_nodes(code2)
            
            if ast_nodes1 and ast_nodes2:  # Only combine if both are not empty
                combined_ast_nodes = ast_nodes1 + ' ' + ast_nodes2
                data.append([combined_ast_nodes, label, ast_nodes1, ast_nodes2])
    
    return pd.DataFrame(data, columns=['code', 'label', 'ast_nodes1', 'ast_nodes2'])

# Function to save preprocessed data with additional features to a file
def save_preprocessed_data(data, output_path):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as necessary
    X = tfidf_vectorizer.fit_transform(data['code'])
    y = data['label']
    
    # Calculate additional features
    manhattan_distances = []
    jaccard_similarities = []

    for _, row in data.iterrows():
        ast_set1 = set(row['ast_nodes1'].split())
        ast_set2 = set(row['ast_nodes2'].split())
        jaccard_similarities.append(jaccard_similarity(ast_set1, ast_set2))
        
        vec1 = tfidf_vectorizer.transform([row['ast_nodes1']]).toarray()[0]
        vec2 = tfidf_vectorizer.transform([row['ast_nodes2']]).toarray()[0]
        manhattan_distances.append(manhattan_distance(vec1, vec2))
    
    # Combine features
    X_additional = np.array([manhattan_distances, jaccard_similarities]).T
    X_combined = np.hstack((X.toarray(), X_additional))
    
    # Apply SMOTE to handle class imbalance
    print(f"Original class distribution: {Counter(y)}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
    print(f"Resampled class distribution: {Counter(y_resampled)}")
    
    joblib.dump((X_resampled, y_resampled, tfidf_vectorizer), output_path)

if __name__ == "__main__":
    java_files_directory = r'C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\8Semestre\PlagiarismDetector\finalDataset\javafiles'
    labels_file_path = r'C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\8Semestre\PlagiarismDetector\finalDataset\unify_labels\javafiles_labels.csv'
    processed_data = preprocess_data_with_ast(java_files_directory, labels_file_path)
    save_preprocessed_data(processed_data, r'C:\Users\Flavio Ruvalcaba\Documents\Escuela\Universidad\8Semestre\PlagiarismDetectorApp\preprocessing\preprocessed_all_data.pkl')
    print("Preprocessing completed and data saved.")
