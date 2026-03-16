import streamlit as st
import pandas as pd
import os

st.title("Company Q&A Annotation Tool (Semantic Chunks)")

# Save progress
ANNOT_SAVE = "qa_annotation_autosave.csv"

# Upload chunked text file
chunk_file = st.file_uploader("Upload chunk_texts.txt", type="txt")

# --- FUNCTION GOES HERE ---
@st.cache_data(show_spinner=False)
def load_chunks(chunk_file):
    data = []
    idx = 1
    for line in chunk_file:
        parts = line.decode("utf-8").strip().split('\t', 1)
        if len(parts) == 2:
            chunk_id, chunk_text = parts
        else:
            chunk_id, chunk_text = str(idx), line.decode("utf-8").strip()
        data.append({"file_chunk": chunk_id, "chunk_text": chunk_text, "question": "", "answer": ""})
        idx += 1
    return pd.DataFrame(data)


if chunk_file:
    df = load_chunks(chunk_file)

    # Load auto-saved progress
    if os.path.exists(ANNOT_SAVE):
        prev = pd.read_csv(ANNOT_SAVE)
        for idx in df.index:
            if idx < len(prev):
                df.loc[idx, 'question'] = prev.loc[idx, 'question']
                df.loc[idx, 'answer'] = prev.loc[idx, 'answer']

    st.write(f"{len(df)} chunks loaded.")
    filter_kw = st.text_input("Filter by keyword (optional)")
    to_annotate = df[df['chunk_text'].str.contains(filter_kw, na=False)] if filter_kw else df

    for idx, row in to_annotate.iterrows():
        st.markdown(f"**[{idx}]** {row['file_chunk']}")
        st.write(row['chunk_text'])
        q = st.text_input(f"Question #{idx}", value=row['question'] or "")
        a = st.text_area(f"Answer #{idx}", value=row['answer'] or "", height=80)
        df.at[idx, 'question'] = q
        df.at[idx, 'answer'] = a
        st.divider()
        if idx % 5 == 4:
            df.to_csv(ANNOT_SAVE, index=False)  # Autosave

    if st.button("Export Reviewed as CSV"):
        df_out = df[(df.question != "") & (df.answer != "")]
        df_out.to_csv("qa_annotation_exported.csv", index=False)
        st.success("Exported to qa_annotation_exported.csv")
        os.remove(ANNOT_SAVE)

# Usage: streamlit run qa_annotation_app.py
