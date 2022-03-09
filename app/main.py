from flask import Flask, render_template, request
import string
from nltk.corpus import stopwords
import pickle
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from nltk import PorterStemmer

app =  Flask(__name__)

def remove_punctuations(text):
    text = [letter for letter in text if letter not in string.punctuation]
    text = ''.join(text)
    text = text.strip()
    return text

def remove_stop_words(text):
    text = [word for word in text.split(' ') if word not in stopwords.words('english')]
    text = ' '.join(text)
    text = text.strip()
    return text

def stem(text):
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text.split(' ')]
    text = ' '.join(text)
    return text

@app.route('/',methods=['POST','GET'])
def similar_cases():
    final_table = None
    if request.method == "POST":
        summary = request.form['summary']
        summary = remove_punctuations(summary)
        summary = remove_stop_words(summary)
        summary = stem(summary)
        
        facts = pickle.load(open('../dependencies/facts_matrix.pkl','rb'))
        vectorizer = pickle.load(open('../dependencies/vectorizer.pkl','rb'))

        summary = vectorizer.transform([summary])
        cosine_similarity = linear_kernel(summary,facts)

        df = pd.read_csv('../dependencies/processed_data.csv')
        df['similarity'] = cosine_similarity.reshape(-1)
        final_table = df.sort_values(by='similarity',ascending=False).iloc[1:10].copy()

    return render_template('cases.html',final_table=final_table)


if __name__ == "__main__":
    app.run(debug=True)