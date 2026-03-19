import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# data
messages = [
    # spam
    "Win a free iPhone now click here",
    "Congratulations you won 1000 dollars",
    "Click this link to claim your prize",
    "Free money waiting for you",
    "You have been selected for a reward",
    "Buy cheap meds online no prescription",
    "Hot singles in your area click now",
    "Earn 500 dollars a day working from home",
    "Urgent your account needs verification click",
    "You won the lottery claim now",
    "Limited offer get rich fast",
    "Free gift card just click here",
    "Double your income guaranteed",
    "Act now limited time offer free cash",
    "Exclusive deal only for you click",

    # ham
    "Hey are you free to meet tomorrow",
    "Can you send me the homework notes",
    "What time does the movie start",
    "I will be late for the meeting",
    "Did you eat lunch already",
    "Let me know when you reach home",
    "The project is due on Friday",
    "Can we reschedule the call to 3pm",
    "I liked the book you recommended",
    "Happy birthday hope you have a great day",
    "The weather looks good today",
    "Please review the document I sent",
    "See you at the office tomorrow",
    "Thanks for your help yesterday",
    "Are you coming to the party tonight",
]

# labels
labels = [
    "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham",
    "ham", "ham", "ham", "ham", "ham",
    "ham", "ham", "ham", "ham", "ham",
]

# pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),  # vectorize
    ("clf", MultinomialNB()),       # classify
])

# train
model.fit(messages, labels)

# save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
