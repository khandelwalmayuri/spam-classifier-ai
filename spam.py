#import the data here 
import pandas as pd 
data=pd.read_csv("sms.tsv",sep="\t",header=None)

data.columns=["label","message"]


data['label_num']=data['label'].map({'ham':0,'spam':1})

x=data['message']
y=data['label_num']

print("data loaded successfully")
print(x.head())
#🧩 STEP 6: Convert text → numbers (Magic step ✨)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
X_vectorized=vectorizer.fit_transform(x)

print("Text converted to numbers!")

#🧩 STEP 7: Train the AI model (REAL AI STARTS)
from sklearn.model_selection import train_test_split
#split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

# Train
model.fit(X_train, y_train)
print("Model trained successfully!")


#Test accuracy
accuracy=model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Test custom message
# msg = ["Hi this is sherya"]#["Congratulations! You won a free prize"]
# msg_vector = vectorizer.transform(msg)
# #testing with a example message
# prediction = model.predict(msg_vector)

# if prediction[0] == 1:
#     print("Spam Message 🚫")
# else:
#     print("Not Spam ✅")

def predict_spam(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    return "Spam 🚫" if prediction[0] == 1 else "Not Spam ✅"
    