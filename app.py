import streamlit as st
import pickle

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered"
)

model = pickle.load(open("fake_review_model.pkl", "rb"))

st.title("🔍 Fake Review Detection")
st.write("AI-powered system to detect **computer-generated vs genuine product reviews**.")

st.info("📊 Model Accuracy: **89.7%**")

st.divider()

review = st.text_area("✍️ Enter a product review to analyze:")

if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        prediction = model.predict([review])
        prob = model.predict_proba([review])

        if prediction[0] == 1:
            confidence = prob[0][1] * 100
            st.error(f"🚨 **Computer Generated Review**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.progress(confidence/100)

        else:
            confidence = prob[0][0] * 100
            st.success(f"✅ **Genuine Human Review**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.progress(confidence/100)

st.divider()

st.caption("Built with Machine Learning using scikit-learn and deployed with Streamlit.")


