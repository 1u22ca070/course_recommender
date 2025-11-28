import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================
# Load Model Artifacts
# ==========================
@st.cache_resource
def load_data():
    courses_df = joblib.load("courses_df.joblib")
    hybrid_sim = np.load("hybrid_sim.npy")
    return courses_df, hybrid_sim

courses_df, hybrid_sim = load_data()

# ==========================
# Recommendation Function
# ==========================
def hybrid_recommend(course_name, n=5):
    course_name = course_name.lower().strip()
    name_to_idx = {c.lower(): i for i, c in enumerate(courses_df['course_name'])}
    
    if course_name not in name_to_idx:
        st.error("❌ Course not found in the dataset.")
        return None

    idx = name_to_idx[course_name]
    sims = hybrid_sim[idx]
    top_indices = np.argsort(sims)[::-1][1:n+1]
    
    recs = courses_df.iloc[top_indices][
        ['course_name', 'instructor', 'difficulty_level', 'rating', 'course_price']
    ].copy()
    recs['Similarity_Score'] = sims[top_indices]
    return recs.reset_index(drop=True)

# ==========================
# Streamlit UI
# ==========================
st.title("🎓 Hybrid Course Recommendation System")

st.markdown("Get personalized course suggestions based on content and collaborative filtering!")

course_list = sorted(courses_df['course_name'].unique())
selected_course = st.selectbox("🔍 Select a Course", course_list)

if st.button("Show Recommendations"):
    results = hybrid_recommend(selected_course, 5)
    if results is not None:
        st.success(f"📚 Top 5 recommended courses similar to **{selected_course}**:")
        st.dataframe(results)
