import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Membaca file training dan test
data_train = pd.read_csv('indonlu/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv', sep='\t', names=["Teks", "Target"])
data_test = pd.read_csv('indonlu/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv', sep='\t', names=["Teks", "Target"])

# Melakukan vektorisasi untuk mengekstrak fitur dengan TF-IDF
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(data_train['Teks'])
test_vectors = vectorizer.transform(data_test['Teks'])

# Load Model
model = joblib.load("./model.joblib")

# Program
os.system("cls")
while True:
    print('==== Deteksi Sentimen ====\n')
    teks = input("Masukkan teks: ")
    teks_vector = vectorizer.transform([teks]) # vectorizing
    print('\n')
    print('===== Keterangan =====\n')
    print(teks,'\n')
    print("Hasil Sentimen: ", model.predict(teks_vector))
    lanjut = input("\nLanjut? (y/n): ")
    if lanjut == 'y':
        os.system("cls")
    elif lanjut == 'n':
        os.system("cls")
        break