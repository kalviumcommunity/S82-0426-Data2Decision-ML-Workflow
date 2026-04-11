import streamlit as st
import pickle
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="Data2Decision AI", 
    page_icon="🤖",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Project Description Analyzer")
st.write("Unlock AI-driven career insights by analyzing your project's technical core.")

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = "models/final_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run the training/serialization step first.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

if model is not None:
    # User input
    st.header("📋 Enter Project Description")
    description = st.text_area(
        "Paste your project README, objective, or technical summary below:",
        placeholder="e.g., A full-stack web application built with React and Python...",
        height=200
    )

    # Predict button
    if st.button("Analyze Project & Generate Insights"):

        if description.strip() == "":
            st.warning("Please enter a project description to proceed.")
        else:
            with st.spinner("Analyzing technical stack..."):
                input_df = pd.DataFrame([{
                    "description": description
                }])

                # Inference
                prediction = model.predict(input_df["description"])
                probs = model.predict_proba(input_df["description"])
                
                # Using 'Python' as the primary example skill based on previous workshops
                is_python = prediction[0] == 1
                confidence = probs[0][1] if is_python else probs[0][0]

                st.divider()

                # Results Layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔍 Detected Core Skills")
                    if is_python:
                        st.success("✅ Python Development")
                    else:
                        st.info("ℹ️ General Tech Stack")
                    st.caption(f"Confidence: {confidence:.2%}")

                with col2:
                    st.subheader("💡 Career Recommendations")
                    if is_python:
                        st.write("• Explore **Machine Learning** or **Data Science** pathways.")
                        st.write("• Build projects using **Flask** or **Django** for backend experience.")
                    else:
                        st.write("• Incorporate **Python** into your workflow to unlock automation.")
                        st.write("• Learn **SQL** to enhance your database management skills.")

                st.divider()
                st.info("💡 **Pro-Tip:** Align your projects with your target company's stack to increase placement chances. 🚀")

else:
    st.info("System is initializing. Please ensure the model is trained and saved.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ for the Data2Decision ML Workflow Workshop")
