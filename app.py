import streamlit as st
import pickle

st.set_page_config(page_title="Fake Review Detector")

model = pickle.load(open("fake_review_model.pkl","rb"))

st.title("🔍 Fake Review Detection")
st.write("AI model to detect fake product reviews")

review = st.text_area("✍️ Enter Review")

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        prediction = model.predict([review])
        prob = model.predict_proba([review])

        if prediction[0] == 1:
            st.error(f"Computer Generated Review/Fake Review (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"Genuine Human Review (Confidence: {prob[0][0]*100:.2f}%)")
