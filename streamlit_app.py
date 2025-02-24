import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

# Load the Iris dataset and train the model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Function to classify iris species
def classify_iris_ml(sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction[0]]

# Function for sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title("Iris Classification and Sentiment Analysis App")

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose an option", ["Iris Classification", "Sentiment Analysis"])

if option == "Iris Classification":
    st.header("Iris Flower Classification")
    
    # Arrange sliders in two columns
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
        petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 3.8)
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)
    
    if st.button("Predict"):  
        prediction = classify_iris_ml(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f"Predicted Iris Species: {prediction}")

elif option == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    user_input = st.text_area("Enter a sentence:")
    
    if st.button("Analyze"): 
        if user_input:
            sentiment = analyze_sentiment(user_input)
            st.success(f"Predicted Sentiment: {sentiment}")
        else:
            st.warning("Please enter some text.")