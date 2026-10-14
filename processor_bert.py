import joblib
from sentence_transformers import SentenceTransformer

clf_model = joblib.load("models/log_classifier.joblib")
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model

def classify_with_bert(log_msg) :
    embeddings = model_embedding.encode([log_msg])
    # Convert from shape (384,) to (1, 384)
    # embeddings = embeddings.reshape(1, -1)
    prediction_probabilities = clf_model.predict_proba(embeddings)[0]

    if max(prediction_probabilities)<0.5:
        return 'unclassified'
    predicted_label = clf_model.predict(embeddings)[0]
    
    return predicted_label


if __name__ == '__main__':
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)