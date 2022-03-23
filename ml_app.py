#from dis import findlinestarts
#from unittest import result
#from soupsieve import escape
import streamlit as st
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
#import random
import pandas as pd
from textblob.classifiers import NaiveBayesClassifier
#from collections import Collections
import nltk

import subprocess
cmd = ['python3','-m','textblob.download_corpora']
subprocess.run(cmd)
print("Working")

#python -m textblob.download_corpora
#https://www.nltk.org/data.html
#nltk.download()

st.title('Natural Language Processing Demo')
st.write('---------------------------------------------------------------------')
st.subheader('Sentiment analysis')

txt = st.text_area('Add English text below for sentiment analaysis (press Ctrl + Enter to apply):', ''' This is an example of a sentence with a very terrible sentiment!''')

def run_sentiment_analysis(asd):
     result = TextBlob(asd).polarity
     return round(result, 2)

st.write('---------------------------------------------------------------------')
st.subheader('Results')
st.write('Sentiment is scored between -1 and +1.    ' + "' - '" + 'suggests negative and ' "' + '" 'positive' + "and " + "' 0 '" + " neutral" + ' sentiment.' + ' See results below: ' )
st.write('Sentiment:', run_sentiment_analysis(txt))

if run_sentiment_analysis(txt) <= -0.5:
     result_Statement = str('*Sentiment is negative!*')
elif run_sentiment_analysis(txt) <= 0:
     result_Statement = str('*Sentiment is neutral or slightly negative*')
elif run_sentiment_analysis(txt) <= 0.5:
     result_Statement = str('*Sentiment is neutral or slightly positive*')
else:
     result_Statement = str('*Sentiment is positive!*')

st.write('', result_Statement)

st.write('---------------------------------------------------------------------')

st.subheader('Transaltion')

option = st.selectbox(
     'Please select language for output translation',
     ('Finnish', 'Swedish', 'German', 'Spanish'))

if option == 'Finnish':
     to_target = 'fi'
elif option == 'Swedish':
     to_target = 'sv'
elif option == 'German':
     to_target = 'de'
else:
     to_target = 'es'

blob = TextBlob(txt)
def translate(asd):
     translation_result = blob.translate(to=to_target)
     return translation_result

st.write('Translation:', translate(blob))
st.write('---------------------------------------------------------------------')

st.subheader('Text classification with Naive Bayes classifier')

train = [
('I love this sandwich.', 'pos'),
('this is an amazing place!', 'pos'),
('I feel very good about these beers.', 'pos'),
('this is my best work.', 'pos'),
("what an awesome view", 'pos'),
('I do not like this restaurant', 'neg'),
('I am tired of this stuff.', 'neg'),
("I can't deal with this", 'neg'),
('he is my sworn enemy!', 'neg'),
('my boss is horrible.', 'neg')
]
test = [
('the beer was good.', 'pos'),
('I do not enjoy my job', 'neg'),
("I ain't feeling dandy today.", 'neg'),
("I feel amazing!", 'pos'),
('Gary is a friend of mine.', 'pos'),
("I can't believe I'm doing this.", 'neg')
]


d = {
     'Phrases':['I love this sandwich.',
     'this is an amazing place!',
     'I feel very good about these beers.',
     'this is my best work.',
     "what an awesome view",
     'I do not like this restaurant',
     'I am tired of this stuff.',
     "I can't deal with this",
     'he is my sworn enemy!',
     'my boss is horrible.',
     'the beer was good.',
     'I do not enjoy my job',
     "I ain't feeling dandy today.",
     "I feel amazing!",
     'Gary is a friend of mine.',
     "I can't believe I'm doing this."],
     'Polarity':['Positive','Positive','Positive','Positive','Positive','Negative','Negative','Negative','Negative','Negative','Positive','Negative','Negative','Positive', 'Positive','Negative']
}

df = pd.DataFrame(data=d)
cl = NaiveBayesClassifier(train)
st.write('This is the model training and testing data used to train the Naive Bayes classifier. Index 0-10 is training data, testing data starts on row #10')
st.table(df)

st.write('---------------------------------------------------------------------')

clas_input = st.text_area('Add any English sentence below for Naive Bayes clasification: ', ''' I really hate this place!''')
clas_input_str = cl.classify(clas_input)


if clas_input_str == 'pos':
     clas_input_result = "This is classified as a positive sentence"
else:
     clas_input_result = "This is classified as a negative sentence"

st.subheader('Results')
st.write('---------------------------------------------------------------------')

st.write("Result: ", clas_input_result)

prob_dist = cl.prob_classify(clas_input)
st.write('The probability of this statement being positive is: ')
st.write(round(prob_dist.prob("pos"), 2))
st.write('The probability of this statement being negative is: ')
st.write(round(prob_dist.prob("neg"), 2))
st.write('Model accuracy: ', round(cl.accuracy(test),2))



# Revome "made with streamlit logo from the bottom"
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 