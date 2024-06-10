import re
import nltk
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Remove URLs, numbers, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    
    # Tokenization and lowercasing
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Lemmatization
    tokens = [nlp(word)[0].lemma_ for word in tokens]
    
    return ' '.join(tokens)

# Example usage
text = "An experienced software developer with a strong background in Python and machine learning."
preprocessed_text = preprocess_text(text)
print(preprocessed_text)
import spacy

nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    doc = nlp(text)
    entities = { "PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": [], "WORK_OF_ART": [] }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

# Example usage
entities = extract_entities(preprocessed_text)
print(entities)
from transformers import pipeline

# Load sentiment-analysis pipeline
sentiment_analysis = pipeline('sentiment-analysis')

# Example usage
sentiment = sentiment_analysis(preprocessed_text)
print(sentiment)

# Dummy personality prediction function (replace with a real model in practice)
def predict_personality(text):
    # Example implementation, in practice use a trained model for prediction
    return {
        "Openness": 0.7,
        "Conscientiousness": 0.6,
        "Extraversion": 0.5,
        "Agreeableness": 0.8,
        "Neuroticism": 0.3
    }

# Example usage
personality_traits = predict_personality(preprocessed_text)
print(personality_traits)
import os
import pandas as pd

def process_cv(cv_text):
    preprocessed_text = preprocess_text(cv_text)
    entities = extract_entities(preprocessed_text)
    sentiment = sentiment_analysis(preprocessed_text)
    personality_traits = predict_personality(preprocessed_text)
    
    return {
        "entities": entities,
        "sentiment": sentiment,
        "personality_traits": personality_traits
    }

def process_all_cvs(cv_directory):
    results = []
    for filename in os.listdir(cv_directory):
        if filename.endswith('.txt'):  # Assuming CVs are in text format
            with open(os.path.join(cv_directory, filename), 'r') as file:
                cv_text = file.read()
                result = process_cv(cv_text)
                result['filename'] = filename
                results.append(result)
    
    return pd.DataFrame(results)

# Example usage
cv_directory = 'path_to_cv_directory'
results_df = process_all_cvs(cv_directory)
print(results_df)
