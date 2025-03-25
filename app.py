
import streamlit as st
import pandas as pd
from datasets import load_dataset
import json
from datetime import datetime
import os
import random

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
    for idx, row in enumerate(image_ds):
        for i, (img1_key, img2_key) in enumerate(pairs):
            template_key = random.choice(list(query_templates.keys()))  # v1 to v5
            var = random.choice(["variant", "invariant"])  # v1 to v5
            condition = f"\n - **{query_conditions['color_jittering'][var]}**\n\n"
            instruction = query_templates[template_key].format(conditions=condition)
            instruction = instruction.replace('Score: <1-10>', '**Score: <1-10>**')

            all_samples.append({
                "dataset": "in100",
                "split": "colorjitter",
                "uid": f"{idx}_{i}",
                "pair": f"[{img1_key}, {img2_key}]",
                "img1": row[img1_key],
                "img2": row[img2_key],
                "var": var,
                "instruction": instruction,
                "template_version": template_key,
            })

    random.shuffle(all_samples)
    return all_samples[:10]

def save_responses(results_df):
    os.makedirs("responses", exist_ok=True)
    path = "responses/user_responses.csv"
    if os.path.exists(path):
        results_df.to_csv(path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(path, index=False)

st.title("Human Evaluation Interface")

template_ds, image_ds = load_data()
samples = prepare_evaluation_samples(template_ds, image_ds)

user_id = st.text_input("Enter your name or ID (optional)", "")

responses = []
for idx, sample in enumerate(samples):
    st.markdown(f"---\n### Sample {idx + 1}")
    st.markdown(f"**Instruction:**\n\n{sample['instruction']}")

    cols = st.columns(2)
    with cols[0]:
        st.image(sample["img1"], caption="Image 1", use_container_width=True)
    with cols[1]:
        st.image(sample["img2"], caption="Image 2", use_container_width=True)

    score = st.slider(f"Your score for Sample {idx + 1}", -1, 10, step=1, key=f"score_{idx}")
    responses.append({
        "user_id": user_id,
        "sample_uid": sample["uid"],
        "instruction_version": sample["template_version"],
        "user_score": score,
        "timestamp": datetime.utcnow().isoformat()
    })

if st.button("✅ Submit All Responses"):
    df = pd.DataFrame(responses)
    save_responses(df)
    st.success("Thanks! Your responses have been recorded.")
