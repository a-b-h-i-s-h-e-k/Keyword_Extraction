import pickle
from flask import Flask, render_template, request  # Flask framework for web app creation and template rendering
import re  # For regular expression operations
import nltk  # For natural language processing
from nltk.stem.wordnet import WordNetLemmatizer  # For lemmatizing words
from nltk.corpus import stopwords  # For removing stopwords

# Initialize Flask application
app = Flask(__name__)

# Load pickled files and preprocessed data
# Why: These files contain pre-trained models and data required for keyword extraction
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)  # Load the CountVectorizer object

with open('tfidf_transformer.pkl', 'rb') as f:
    tfidf_transformer = pickle.load(f)  # Load the TF-IDF Transformer object

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)  # Load the list of feature names (vocabulary)

# Cleaning data: Extend stopwords with custom words
# Why: To exclude irrelevant or frequently occurring words that don't add value to keyword extraction
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using", 
                  "show", "result", "large", "also", "one", 
                  "two", "three", "four", "five", "seven", 
                  "eight", "nine"]
stop_words = list(stop_words.union(new_stop_words))  # Combine default and custom stopwords

# Function to preprocess input text
# Why: Prepares raw text for keyword extraction by cleaning, tokenizing, and lemmatizing
def preprocess_text(txt):
    # Lowercase the text
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # Tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    txt = [word for word in txt if word not in stop_words]
    # Remove short words (less than 3 characters)
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatization
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

# Function to sort non-zero elements of a sparse matrix
# Why: Orders terms by their scores in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# Function to extract top `n` keywords from a sorted TF-IDF vector
# Why: Provides the highest-ranking terms based on their TF-IDF scores
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]  # Limit to top `n` items

    score_vals = []  # To store the scores of the keywords
    feature_vals = []  # To store the corresponding feature names (keywords)
    
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))  # Round scores to 3 decimals
        feature_vals.append(fname)

    # Create a dictionary of keywords and their scores
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

# Route for the home page
@app.route('/')
def index():
    # Renders the main index page (upload and search options)
    return render_template('index.html')

# Route to handle document upload and extract keywords
@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']  # Get the uploaded file
    if document.filename == '':
        # If no file is selected, return to the main page with an error
        return render_template('index.html', error='No document selected')

    if document:
        # Read and decode the file contents
        text = document.read().decode('utf-8', errors='ignore')
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        # Transform the preprocessed text into a TF-IDF vector
        tf_idf_vector = tfidf_transformer.transform(cv.transform([preprocessed_text]))
        # Sort the TF-IDF vector by scores
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        # Extract top keywords
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        # Render the results page with the extracted keywords
        return render_template('keywords.html', keywords=keywords)
    return render_template('index.html')

# Route to handle keyword search
@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']  # Get the search query
    if search_query:
        # Search for keywords containing the query
        keywords = []
        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                # Limit to 20 results
                if len(keywords) == 20:
                    break
        # Render the results page with the search results
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)
