# Keyword Extraction Application
Overview

- This project is a web-based application that extracts and displays keywords from uploaded documents. It is built using Python Flask for the backend and HTML with Bootstrap for the frontend. The application also allows users to search for specific keywords within the dataset.

-> Features

1. Keyword Extraction:
- Upload documents (e.g., .txt or .pdf converted to text).
- Extracts the most relevant keywords using TF-IDF.
- Displays keywords along with their importance scores.

2. Keyword Search:
- Search for specific keywords in the dataset.
- Displays matching keywords in a user-friendly interface.

3. Frontend:
- Uses Bootstrap for a clean, responsive UI.
- Displays results in a table or as a styled keyword list.

4.Backend:
- Flask framework handles routes, file processing, and keyword extraction.
- Pretrained models for vectorization and transformation are loaded from serialized pickle files.

## Installation
#### Prerequisites
- Python 3.x
- Flask
- NLTK
- scikit-learn
- Bootstrap (integrated via CDN)

File Structure
Backend
app.py:

Contains the Flask application logic.
Routes for uploading documents, extracting keywords, and searching keywords.
count_vectorizer.pkl:

Pretrained CountVectorizer object for vectorizing text.
tfidf_transformer.pkl:

Pretrained TF-IDF transformer for generating scores.
feature_names.pkl:

Contains the vocabulary of feature names.
Frontend
templates/index.html:

The main page with forms for uploading documents and searching keywords.
templates/keywords.html:

Displays extracted keywords with their scores in a table.
templates/keywordslist.html:

Shows a list of matching keywords for search queries.
Others
Keyword_extraction.ipynb:
Jupyter Notebook for model development and testing.
Usage
1. Extract Keywords
Navigate to the Upload Document section on the homepage.
Upload a text document.
Click Extract Keywords to view the extracted keywords and their scores.
2. Search Keywords
Enter a keyword or phrase in the search bar.
Click Search to display matching keywords from the dataset.


Customization
Adjust Stopwords
To customize stopwords, modify the new_stop_words list in app.py:

new_stop_words = ["fig", "figure", "image", "sample", "using",
                  "show", "result", "large", "also", "one"]

Modify Top N Keywords
To change the number of keywords displayed, update the topn parameter in extract_topn_from_vector:

def extract_topn_from_vector(feature_names, sorted_items, topn=20):
    # Change 20 to your desired number
Credits
Bootstrap: For responsive design.
NLTK: For text preprocessing.
scikit-learn: For TF-IDF and vectorization.
