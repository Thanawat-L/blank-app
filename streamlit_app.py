import streamlit as st
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError
import json
import os

st.set_page_config(
    page_title="Machine Learning for MADT application",
    page_icon="🌟",
    layout="wide" 
)

st.title("🌟 Machine Learning for MADT application")

# ---------------- Upload JSON file ----------------
def test_bigquery_key(json_file_path):
    try:
        client = bigquery.Client.from_service_account_json(json_file_path)
        
        project = client.project
        st.success(f"BigQuery key is valid! Connected to project: {project}")
        return True
    except GoogleAPICallError as e:
        st.error(f"BigQuery key is invalid. Error: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False

json_file = st.file_uploader("Upload a JSON file", type=["json"])

if json_file is not None:
    # Templorary create JSON file for test
    temp_file_path = "temp_key.json"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(json_file.read())
    
    # key testing
    if test_bigquery_key(temp_file_path):
        st.write("Key is ready to use!")
    else:
        st.write("Please check your key file.")
    
    # remove temp .json file
    os.remove(temp_file_path)

# ---------------- Button to run ML ----------------
if "ml_run" not in st.session_state:
    st.session_state.ml_run = False

if st.button("Run ML"):

    # Coding

    st.session_state.ml_run = True

