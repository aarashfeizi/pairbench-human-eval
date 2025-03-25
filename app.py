
import streamlit as st
import pandas as pd
from datasets import load_dataset
import json
from datetime import datetime
import os
import random
import gspread
from google.oauth2.service_account import Credentials

constant_template = """
Score the similarity of the two images **on a scale of 1 (least similar) to 10 (completely similar)** given the condition[s] below: \n {conditions} \n 

***Note: Identical images that do not meet the condtion[s] should still score more than irrelevant images.***

"""

def write_to_gsheet(data):
    creds_dict = st.secrets["gcp_service_account"]
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)

    spreadsheet_id = "13bTYTcnvslTKc_fJIun-w-gpDOgSeTluAWDrgmXysng"
    sheet = client.open_by_key(spreadsheet_id).sheet1

    columns = [
        "user_id", "row_number", "sample_uid", "instruction_version", "instruction", "user_score",
        "timestamp", "dataset", "split", "pair", "var"
    ]

    if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
        sheet.append_row(columns)

    for row in data:
        sheet.append_row([row.get(col, "") for col in columns])

@st.cache_data
def load_data():
    template_ds = load_dataset("feiziaarash/mmscore", name="templates", split="in100[:100]")
    image_ds = load_dataset("feiziaarash/mmscore", name="in100", split="colorjitter[:100]")
    return template_ds, image_ds

def prepare_evaluation_samples(template_ds, image_ds):
    template = template_ds[0]
    query_templates = json.loads(template["query_templates"])
    query_conditions = json.loads(template["query_conditions"])
    logistics = json.loads(template["logistics"])
    pairs = logistics["data-pairs"]

    all_samples = []
    # rnd = random.Random(hash(st.session_state.user_id))
    rnd = random.Random() 
    for idx, row in enumerate(image_ds):
        for i, (img1_key, img2_key) in enumerate(pairs):
            template_key = rnd.choice(list(query_templates.keys()))
            var = rnd.choice(["variant", "invariant"])
            condition = f"\n - **{query_conditions['color_jittering'][var]}**\n\n"
            instruction = constant_template.format(conditions=condition)
            # instruction = query_templates[template_key].format(conditions=condition)
            instruction = instruction.replace('Score: <1-10>', '**Score: <1-10>**')

            all_samples.append({
                "dataset": "in100",
                "split": "colorjitter",
                "uid": f"{idx}_{i}",
                "row_number": f"{idx}",
                "pair": f"{img1_key}-{img2_key}",
                "img1": row[img1_key],
                "img2": row[img2_key],
                "var": var,
                "instruction": instruction,
                # "template_version": template_key,
                "template_version": 'const',
            })

    rnd.shuffle(all_samples)
    return all_samples[:20]

st.title("PairBench Human Evaluation")

if "samples" not in st.session_state:
    template_ds, image_ds = load_data()
    st.session_state.samples = prepare_evaluation_samples(template_ds, image_ds)
    st.session_state.current_sample_idx = 0
    st.session_state.responses = {}

samples = st.session_state.samples
sample_idx = st.session_state.current_sample_idx
sample = samples[sample_idx]

if "user_id" not in st.session_state:
    user_input = st.text_input("Enter your name or ID (required):", "")
    if user_input:
        st.session_state.user_id = user_input
        st.rerun()
    else:
        st.warning("You must enter a user ID to begin.")
        st.stop()
else:
    st.markdown(f"👤 **User ID:** `{st.session_state.user_id}`")    
    st.warning(f"**Do not refresh page while taking the survey!**")    
        
st.markdown(f"---\n### Sample {sample_idx + 1} of {len(samples)}")
st.markdown(f"**Instruction:**\n\n{sample['instruction']}")

cols = st.columns(2)
with cols[0]:
    st.image(sample["img1"], caption="Image 1", use_container_width=True)
with cols[1]:
    st.image(sample["img2"], caption="Image 2", use_container_width=True)

score_key = f"score_{sample['uid']}"
default_score = st.session_state.responses.get(sample['uid'], {}).get("user_score", -1)

st.markdown("**Select your score (1 = low similarity, 10 = high similarity):**")

# Show previous score if available
previous_score = st.session_state.responses.get(sample['uid'], {}).get("user_score")
if previous_score:
    st.markdown(f"🔁 You previously selected: **{previous_score}**")


score_col = st.columns(10)
for i, col in enumerate(score_col, start=1):
    with col:
        if st.button(str(i), key=f"score_btn_{sample['uid']}_{i}"):
            st.session_state.responses[sample['uid']] = {
                "user_id": st.session_state.user_id,
                "row_number": sample['row_number'],
                "sample_uid": sample["uid"],
                "instruction_version": sample["template_version"],
                "instruction": sample["instruction"],
                "user_score": i,
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": sample['dataset'],
                "split": sample['split'],
                "pair": sample['pair'],
                "var": sample["var"]
            }

            if sample_idx < len(samples) - 1:
                st.session_state.current_sample_idx += 1
                st.rerun()

# Back + Restart side-by-side
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if sample_idx > 0:
        if st.button("⬅️ Back"):
            st.session_state.current_sample_idx -= 1
            st.rerun()
with col2:
    # Show skip button only if score already exists
    if previous_score and sample_idx < len(samples) - 1:
        if st.button("⏭️ Skip to next sample", key=f"skip_{sample['uid']}"):
            st.session_state.current_sample_idx += 1
            st.rerun()
        
with col3:
    if st.button("🔁 Restart with new samples"):
        template_ds, image_ds = load_data()
        # Use a new seed for fresh randomness on restart
        st.session_state.samples = prepare_evaluation_samples(template_ds, image_ds)
        st.session_state.current_sample_idx = 0
        st.session_state.responses = {}
        st.session_state.submitted = False
        st.rerun()    
        
progress = (sample_idx + 1) / len(samples)
st.markdown("---")
st.progress(progress, text=f"Progress: {sample_idx + 1} / {len(samples)}")

# Final submission logic
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Submission screen (last sample)
if sample_idx == len(samples) - 1:
    if not st.session_state.submitted:
        st.markdown("---")
        st.markdown("### 🎯 Final Step: Submit Your Responses")
        st.markdown("Please make sure you've reviewed everything before submitting.")

        if st.button("✅ Submit All Responses"):
            with st.container():
                with st.spinner("🌀 Submitting your responses... Please wait."):
                    # This simulates a submission delay (optional, remove if not needed)
                    response_list = list(st.session_state.responses.values())
                    df = pd.DataFrame(response_list)
                    write_to_gsheet(df.to_dict(orient="records"))
                    st.session_state.submitted = True
                    st.rerun()  # Rerun to show post-submit UI
    else:
        # Grayed out overlay (simple trick: show only this section, no back, no buttons)
        st.markdown("""
        <div style='
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 9990;
        '></div>
        """, unsafe_allow_html=True)

        st.markdown("## ✅ Your responses have been submitted.")
        st.success("✅ Thank you! Your responses have been recorded.")
        st.info("🔒 You may now close the tab or refresh the page to restart.")
        st.stop()