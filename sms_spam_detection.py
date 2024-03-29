# -*- coding: utf-8 -*-
"""SMS_Spam_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BaaWm5ERPClE53qWWCq0HPn1IjMPUKr5
"""

import numpy as np
import pandas as pd

# Try reading the CSV file with different encodings
try:
    df = pd.read_csv("spam.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("spam.csv", encoding='ISO-8859-1')
    except UnicodeDecodeError:
        df = pd.read_csv("spam.csv", encoding='Windows-1252')

df.sample(5)

df.shape

"""## Data Cleaning-"""

df.info()

# Dropping the last 3 columns-
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

df.sample(5)

# renaming the columns-
df.rename(columns={'v1': 'target', 'v2': 'text'},inplace = True)
df.sample(5)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

# 0 = Ham and 1 = Spam
df.head()

# Checking for missing values-
df.isnull().sum()

# Checking for duplicate values-
df.duplicated().sum()

# Removing duplicates-
df = df.drop_duplicates(keep = 'first')

# Checking for duplicate values again-
df.duplicated().sum()

df.shape

"""EDA (EXPLORATORY DATA ANALYSIS)-"""

df.head()

# Determining value counts of spam and ham values-
df['target'].value_counts()

# Creating Visual Representation of the data-
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
plt.show()

import nltk
nltk.download('punkt')

df['num_characters'] = df['text'].apply(len)

df.head()

# Fetching the number of words using nltk-
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x))) # - This code breaks down every character of a message into a list and counts them too

df.head()

# Creating a column for number of sentences in each sms-
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

df.head()

# Desscribing the statistics of all messages-
df[['num_characters', 'num_words', 'num_sentences']].describe()

# Showing Stats for only Ham Messages-
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()

# Showing the stats for only Spam messages-
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()

# Plotting histograms-
import seaborn as sns

# Extracting and creating histogram for Ham Messages(0) and Spam Messages(1)-
plt.figure(figsize=(10,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='red')

# Plotting histograms for spam and ham messages based on number of words per message-
plt.figure(figsize=(10,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='red')

# Creating Visualization for exploring the Relationship between these variables-
sns.pairplot(df,hue='target')

# Looking for correlations using heatmap-
sns.heatmap(df.corr(), annot=True)

"""# Data Preprocessing-
consists of-


converting data to lowercase


tokenization

removing special characters

removing stopwords and punctuation

stemming

"""

# Downloading and importing stopwords and punctuation marks. Also Importing libraries for Stemming-
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
  text = text.lower() # For Converting to lowercase
  text = nltk.word_tokenize(text)  # For Tokenization

  y = []                      # For removing special characters
  for i in text:              # For Loop that says for each item in the text,
    if i.isalnum():           # If the item is an alphanumeric character
      y.append(i)             # Append the item to the list called y


  text = y[:]                 # Loop for removing stopwords and punctuation
  y.clear()

  for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
          y.append(i)

  text = y[:]                 # Loop for Stemming
  y.clear()

  for i in text:
      y.append(ps.stem(i))

  return " ".join(y)                    # prints the output value of y in form of a string

transform_text('I loved the YT lectures on Machine Learning. How about you?')

df['transformed_text'] = df['text'].apply(transform_text)            # Transforming the text

df.head()

# Generating WordCloud-
from wordcloud import WordCloud      # Importing WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='White')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))
#This line creates a word cloud (`spam_wc`) from the transformed text of rows for spam messages.

# Displaying the wordcloud-
plt.imshow(spam_wc)

# Generating wordcloud for ham messages-
ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))

# Displaying the wordcloud-
plt.imshow(ham_wc)

df.head()

spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():       # converts all the spam texts from transformed_text column to a list
    for word in msg.split():
      spam_corpus.append(word)

len(spam_corpus)

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():       # converts all the ham texts from transformed_text column to a list
    for word in msg.split():
      ham_corpus.append(word)

len(ham_corpus)

"""# Model Building-"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer     # Importing count vectorizer and TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = df['target'].values

y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB        # Importing Naive Bayes
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score   # Importing metrics from sklearn's metrics library

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# This code will print the output of Gaussian Naive Bayes -
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))

# This code will print the output of Multinomial Naive Bayes-
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))

bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))

"""# Chose Tfidf with MNB"""

# Importing useful libraries-
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid',gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l2')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'KN'  : knc,
    'NB'  : mnb,
    'DT'  : dtc,
    'LR'  : lrc,
    'RF'  : rfc,
    'AdaBoost' : abc,
    'BgC'      : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'xgb' : xgb
}

def train_classifier(clf,X_train,y_train,X_test,y_test):
  clf.fit(X_train,y_train)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  precision = precision_score(y_test,y_pred)

  return accuracy, precision

train_classifier(svc,X_train,y_train,X_test,y_test)

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():

    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)

    print("For ",name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(), 'Accuracy':accuracy_scores, 'Precision':precision_scores}).sort_values('Precision',ascending=False)

performance_df

performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")

performance_df1

"""# Trying to Improve The Model-"""

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)

new_df = performance_df.merge(temp_df,on='Algorithm')

new_df_scaled = new_df.merge(temp_df,on='Algorithm')

temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)

new_df_scaled.merge(temp_df,on='Algorithm')

from sklearn.ensemble import VotingClassifier
# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()

from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define the preprocessing function
def preprocess_message(message):
    # Convert to lowercase
    message = message.lower()
    # Tokenization
    message = nltk.word_tokenize(message)
    # Remove special characters
    message = [word for word in message if word.isalnum()]
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    message = [word for word in message if word not in stop_words and word not in string.punctuation]
    # Stemming
    porter = PorterStemmer()
    message = [porter.stem(word) for word in message]
    return " ".join(message)

# Example SMS message
sms_message = "Enter The message that you want to check here."

# Preprocess the SMS message
preprocessed_message = preprocess_message(sms_message)

# Transform the preprocessed message using TfidfVectorizer
message_vector = tfidf.transform([preprocessed_message])

# Convert sparse matrix to dense numpy array
message_vector_dense = message_vector.toarray()

# Scale the message vector
message_vector_scaled = scaler.transform(message_vector_dense)

# Predict whether the message is spam or not spam
prediction = voting.predict(message_vector_scaled)

# Display the prediction
if prediction == 1:
    print("The message is classified as spam.")
else:
    print("The message is not spam.")