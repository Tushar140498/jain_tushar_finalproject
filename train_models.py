"""
train_models.py
In this python file we are tarining 3 models with 10-fold cross-validation:
- RandomForestClassifier (tabular, using TF-IDF features)
- Multinomial Naive Bayes (traditional, using Count features)
- Bidirectional LSTM (deep learning, using Keras Tokenizer & Embedding)

Outputs per-fold metrics and saves models/plots to outputs/
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import joblib
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

DATA_PATH = "data/sms_spam.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df['text_clean'].astype(str)
y = df['label'].astype(int)

# Vectorizers
count_vec = CountVectorizer()
tfidf_vec = TfidfVectorizer()

X_count = count_vec.fit_transform(X)
X_tfidf = tfidf_vec.fit_transform(X)

# Save vectorizers
joblib.dump(count_vec, os.path.join(OUT_DIR, "count_vectorizer.joblib"))
joblib.dump(tfidf_vec, os.path.join(OUT_DIR, "tfidf_vectorizer.joblib"))

# Prepare tokenizer for Bi-LSTM
MAX_WORDS = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
joblib.dump(tokenizer, os.path.join(OUT_DIR, "tokenizer.joblib"))
sequences = tokenizer.texts_to_sequences(X)
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)

# K-fold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results = {
    "rf": [],
    "nb": [],
    "bilstm": []
}

fold = 0
for train_index, test_index in kf.split(X, y):
    fold += 1
    print(f"Fold {fold}")
    X_train_count, X_test_count = X_count[train_index], X_count[test_index]
    X_train_tfidf, X_test_tfidf = X_tfidf[train_index], X_tfidf[test_index]
    X_train_seq, X_test_seq = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #  Random Forest (using TF-IDF)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)
    y_pred_rf = rf.predict(X_test_tfidf)
    y_prob_rf = rf.predict_proba(X_test_tfidf)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    results['rf'].append({"fold":fold,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"auc":float(auc_rf)})
    joblib.dump(rf, os.path.join(OUT_DIR, f"rf_fold{fold}.joblib"))

    # Naive Bayes (using Count features)
    nb = MultinomialNB()
    nb.fit(X_train_count, y_train)
    y_pred_nb = nb.predict(X_test_count)
    y_prob_nb = nb.predict_proba(X_test_count)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_nb).ravel()
    auc_nb = roc_auc_score(y_test, y_prob_nb)
    results['nb'].append({"fold":fold,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"auc":float(auc_nb)})
    joblib.dump(nb, os.path.join(OUT_DIR, f"nb_fold{fold}.joblib"))

    # Bi-LSTM (Keras)
    # Build a small Bi-LSTM model
    EMBEDDING_DIM = 100
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    model.fit(X_train_seq, y_train, validation_split=0.1, epochs=8, batch_size=64, callbacks=[early], verbose=0)
    y_prob_bilstm = model.predict(X_test_seq).flatten()
    y_pred_bilstm = (y_prob_bilstm >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bilstm).ravel()
    auc_bilstm = roc_auc_score(y_test, y_prob_bilstm)
    results['bilstm'].append({"fold":fold,"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"auc":float(auc_bilstm)})
    model.save(os.path.join(OUT_DIR, f"bilstm_fold{fold}.keras"))

    # ROC plot per model per fold (saved)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
    fpr_bi, tpr_bi, _ = roc_curve(y_test, y_prob_bilstm)
    plt.figure(figsize=(6,4))
    plt.plot(fpr_rf, tpr_rf, label=f'RF AUC={auc_rf:.3f}')
    plt.plot(fpr_nb, tpr_nb, label=f'NB AUC={auc_nb:.3f}')
    plt.plot(fpr_bi, tpr_bi, label=f'BiLSTM AUC={auc_bilstm:.3f}')
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Fold {fold}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'roc_fold_{fold}.png'))
    plt.close()

# Save results JSON
with open(os.path.join(OUT_DIR, "cv_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Done. Saved outputs to", OUT_DIR)
