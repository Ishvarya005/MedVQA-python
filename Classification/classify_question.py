import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def find_similar_question(user_input, questions):
    processed_input = process_text(user_input)
    
    processed_questions = [process_text(question) for question in questions]
    
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform([processed_input] + processed_questions)
    
    similarities = cosine_similarity(question_vectors[0], question_vectors[1:])[0]
    
    most_similar_index = similarities.argmax()
    
    return questions[most_similar_index]


if __name__ == "__main__":
    with open("questions.txt", "r") as file:
        questions = file.readlines()

    user_input = input("Enter your question: ")

    similar_question = find_similar_question(user_input, questions)
    print("Similar question:", similar_question)