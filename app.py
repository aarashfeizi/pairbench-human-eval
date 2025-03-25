import streamlit as st
import pandas as pd
from datasets import load_dataset
import json
from datetime import datetime
import random
import gspread
from google.oauth2.service_account import Credentials

constant_template = """
Score the similarity of the two images **on a scale of 1 (least similar) to 10 (completely similar)** given the condition[s] below: \n {conditions} \n 

*Note: Identical images that do not meet the condtion[s] should score **higher** than irrelevant images.*
"""

split2condition = {
    'rotate': "rotation",
    'colorjitter': "color_jittering",
    'perspective': 'perspective'
}

def write_to_gsheet(data):
    creds_dict = st.secrets["gcp_service_account"]
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet_id = "13bTYTcnvslTKc_fJIun-w-gpDOgSeTluAWDrgmXysng"
    sheet = client.open_by_key(spreadsheet_id).sheet1
    columns = ["user_id", "row_number", "sample_uid", "instruction_version", "instruction", "user_score",
               "timestamp", "dataset", "split", "pair", "var"]
    if sheet.row_count == 0 or sheet.cell(1, 1).value is None:
        sheet.append_row(columns)
    for row in data:
        sheet.append_row([row.get(col, "") for col in columns])

# @st.cache_data
def load_data():
    rnd = random.Random()
    split = rnd.choice(["colorjitter", "rotate", "perspective"])
    template_ds = load_dataset("feiziaarash/mmscore", name="templates", split="in100[:100]")
    image_ds = load_dataset("feiziaarash/mmscore", name="in100", split=f"{split}[:100]")
    return template_ds, image_ds, split

def prepare_evaluation_samples(template_ds, image_ds, split_name):
    template = template_ds[0]
    query_templates = json.loads(template["query_templates"])
    query_conditions = json.loads(template["query_conditions"])
    logistics = json.loads(template["logistics"])
    pairs = logistics["data-pairs"]
    all_samples = []
    rnd = random.Random()
    for idx, row in enumerate(image_ds):
        for i, (img1_key, img2_key) in enumerate(pairs):
            template_key = rnd.choice(list(query_templates.keys()))
            var = rnd.choice(["variant", "invariant"])
            condition = f"\n - **{query_conditions[split2condition[split_name]][var]}**\n\n"
            instruction = constant_template.format(conditions=condition)
            instruction = instruction.replace('Score: <1-10>', '**Score: <1-10>**')
            all_samples.append({
                "dataset": "in100",
                "split": split_name,  # Record the split used
                "uid": f"{idx}_{i}",
                "row_number": f"{idx}",
                "pair": f"{img1_key}-{img2_key}",
                "img1": row[img1_key],
                "img2": row[img2_key],
                "var": var,
                "instruction": instruction,
                "template_version": 'const',
            })
    rnd.shuffle(all_samples)
    return all_samples[:20]

st.title("PairBench Human Evaluation")

if "samples" not in st.session_state:
    template_ds, image_ds, split_name = load_data()
    st.session_state.samples = prepare_evaluation_samples(template_ds, image_ds, split_name)
    st.session_state.current_sample_idx = 0
    st.session_state.responses = {}
    st.session_state.submitted = False

samples = st.session_state.samples
sample_idx = st.session_state.current_sample_idx
sample = samples[sample_idx]
is_last_sample = sample_idx == len(samples) - 1

if "user_id" not in st.session_state:
    st.markdown("🖥️ *Use a desktop browser for best experience*")
    st.markdown("### 📝 Quick Survey: Scoring Image Pairs")
    st.markdown(
        "Thank you for participating in this short human evaluation! "
        "You'll be shown **20 image pairs**, each with a specific instruction. "
        "Your task is to assign a **similarity score** based on the condition provided."
    )

    st.info("""
    👀 **Before you begin:**

    ⏱️ This should take **just a few minutes** — so please stay focused, and let's begin!

    You’ll be rating the **similarity** between image pairs.

    There are **20 pairs** to score. The instruction is always similar,  
    but the **condition changes per pair** — so please read each one carefully.

    ✅ **Your job is to follow the condition and give a similarity score from 1 (least similar) to 10 (highest similarity).**

    Each instruction includes one of the following **conditions**:

    - 🔴 **Variant**: Be sensitive to changes — small variations in **color**, **rotation**, or **perspective** should **lower** your similarity score.

    - 🔵 **Invariant**: Ignore such changes — differences in **color**, **rotation**, or **perspective** should **not affect** your similarity score.

    ---

    ❗ **Important:** Your progress will **not** be saved automatically.

    👉 You **must** click the **✅ Submit All Responses** button at the end of the survey  
    to make sure your responses are recorded.

    If you close or refresh the tab before submitting, **your progress will be lost**.
    """)

    user_input = st.text_input("Enter a nickname for yourself (required) and press continue to proceed:", key="user_id_input")
    submit_id = st.button("➡️ Continue")
    if submit_id:
        if user_input.strip():
            st.session_state.user_id = user_input.strip()
            st.rerun()
        else:
            st.warning("You must enter a nickname to begin.")
            st.stop()
    else:
        st.stop()
else:
    st.markdown(f"👤 **User ID:** `{st.session_state.user_id}`")
    st.warning(f"**Do not refresh page while taking the survey!**")

if not is_last_sample:
    st.markdown(f"---\n### Sample {sample_idx + 1} of {len(samples)}")
    # st.markdown(f"---\n### Sample {sample_idx + 1} of {len(samples)}")
    st.markdown(f"**Instruction:**\n\n{sample['instruction']}")
    
    layout = st.columns([1, 1, 1])  # Wider left column for instruction

    with layout[0]:
        template = "**TL;DR Conditions:** \n - **{var} to {split}**"
        st.markdown(f"{template.format(var=sample['var'].upper(), split=sample['split'])}")

    with layout[1]:
        st.image(sample["img1"], caption="Image 1", use_container_width=True)

    with layout[2]:
        st.image(sample["img2"], caption="Image 2", use_container_width=True)

    # cols = st.columns(2)
    # with cols[0]:
    #     st.image(sample["img1"], caption="Image 1", use_container_width=True)
    # with cols[1]:
    #     st.image(sample["img2"], caption="Image 2", use_container_width=True)
    st.markdown("**Select your score (1 = low similarity, 10 = high similarity):**")
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

    # Buttons: Back / Skip / Restart
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if sample_idx > 0 and st.button("⬅️ Back"):
            st.session_state.current_sample_idx -= 1
            st.rerun()
    with col2:
        if previous_score and sample_idx < len(samples) - 1:
            if st.button("⏭️ Skip to next sample", key=f"skip_{sample['uid']}"):
                st.session_state.current_sample_idx += 1
                st.rerun()
    with col3:
        if st.button("🔁 Restart with new samples"):
            template_ds, image_ds, split_name = load_data()
            st.session_state.samples = prepare_evaluation_samples(template_ds, image_ds, split_name)
            st.session_state.current_sample_idx = 0
            st.session_state.responses = {}
            st.session_state.submitted = False
            st.rerun()

    # Progress bar
    progress = (sample_idx + 1) / len(samples)
    st.markdown("---")
    st.progress(progress, text=f"Progress: {sample_idx + 1} / {len(samples)}")

else:
    st.markdown("## 🎯 Final Step: Submit Your Responses")
    st.warning("Please confirm you've completed all evaluations. You can go back and revise if needed.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.current_sample_idx -= 1
            st.rerun()
    with col2:
        if not st.session_state.get("submitted", False):
            if st.button("✅ Submit All Responses"):
                with st.spinner("🌀 Submitting your responses..."):
                    df = pd.DataFrame(list(st.session_state.responses.values()))
                    write_to_gsheet(df.to_dict(orient="records"))
                    st.session_state.submitted = True
                    st.rerun()
        else:
            st.success("✅ Responses already submitted.")
            st.info("🔒 You may now close the tab or restart.")
            st.stop()
