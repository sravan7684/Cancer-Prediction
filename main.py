import streamlit as st

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = None

# Main page
st.title("Cancer Detection")

# Brain Tumor Detection
if st.button("Upload your files for Brain Tumor Detection"):
    st.session_state.page = "brain_tumor"
    st.experimental_rerun()

# Cancer Detection
if st.button("Upload your files for Cancer Detection"):
    st.session_state.page = "cancer"
    st.experimental_rerun()

# Handle redirection based on session state
if st.session_state.page == "brain_tumor":
    st.write("Redirecting to Brain Tumor Detection page...")
elif st.session_state.page == "cancer":
    st.write("Redirecting to Cancer Detection page...")