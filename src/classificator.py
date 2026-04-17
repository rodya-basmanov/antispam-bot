import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

# Настройка путей относительно корня проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
VECTORIZER_FILE = os.path.join(BASE_DIR, 'models', 'vectorizer.joblib')
DATA_FILE = os.path.join(BASE_DIR, 'data', 'data.txt')

# Загрузка ресурсов NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('russian'))

morph = MorphAnalyzer()

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            print("Классификатор: Модель и векторизатор загружены.")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
    else:
        print(f"Файлы модели не найдены по путям:\n{MODEL_FILE}\n{VECTORIZER_FILE}")

# Пробуем загрузить при импорте
load_model()

def classify_message(message):
    """
    Возвращает True, если сообщение - спам.
    """
    if model is None or vectorizer is None:
        print("Ошибка: Модель не загружена!")
        return False

    clean_message = preprocess_text(message)
    message_tfidf = vectorizer.transform([clean_message])
    prediction = model.predict(message_tfidf)

    return bool(prediction[0] == 1)

def preprocess_text(text):
    """
    Функция для предобработки текста.
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URL
    text = re.sub(r'[^\w\s]', '', text)  # Знаки препинания
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [morph.parse(token)[0].normal_form for token in tokens if token not in stop_words]

    return ' '.join(lemmatized_tokens)

def train():
    if not os.path.exists(DATA_FILE):
        print(f"Ошибка: Файл {DATA_FILE} не найден.")
        return 0

    with open(DATA_FILE, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            label, message = parts
            data.append({"label": label, "message": message})

    if not data:
        print("Ошибка: Данные в data.txt отсутствуют или неверно отформатированы.")
        return 0

    df = pd.DataFrame(data)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['processed_text'] = df['message'].apply(preprocess_text)

    X = df['processed_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer_new = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.9,
        max_features=10000
    )

    X_train_vec = vectorizer_new.fit_transform(X_train)
    X_test_vec = vectorizer_new.transform(X_test)

    lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    lr_grid = GridSearchCV(LogisticRegression(), param_grid=lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
    lr_grid.fit(X_train_vec, y_train)

    best_clf = lr_grid.best_estimator_

    preds = best_clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- Результаты ---")
    print(f"Accuracy: {acc:.4f}")
    
    # Создаем папки если нет
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    
    joblib.dump(best_clf, MODEL_FILE)
    joblib.dump(vectorizer_new, VECTORIZER_FILE)
    
    # Обновляем глобальные переменные
    global model, vectorizer
    model = best_clf
    vectorizer = vectorizer_new
    
    print(f"Модель сохранена в {MODEL_FILE}")
    print(f"Векторизатор сохранен в {VECTORIZER_FILE}")

    return acc

if __name__ == "__main__":
    train()
