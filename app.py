
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="DecideX-TOPSIS", layout="wide")

st.title("DecideX-TOPSIS: Smarter Ranking for Innovation Impact")
st.subheader("Multi-Criteria Decision Making using TOPSIS")

st.markdown("Upload your decision matrix and define weights & criteria types to get rankings.")

uploaded_file = st.file_uploader("Upload CSV (first column = Alternatives, first row = Criteria)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.write("### Decision Matrix")
    st.dataframe(df)

    criteria = df.columns.tolist()

    weights = st.text_input(f"Enter weights for criteria (comma-separated) [total should sum to 1]:", value=",".join(["0.2"]*len(criteria)))
    types = st.text_input(f"Enter criteria types (1=Benefit, 0=Cost) comma-separated:", value=",".join(["1"]*len(criteria)))

    if st.button("Run TOPSIS"):
        try:
            weights = np.array([float(w) for w in weights.split(",")])
            types = np.array([int(t) for t in types.split(",")])
            matrix = df.values.astype(float)

            # Step 1: Normalize matrix
            norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

            # Step 2: Weight normalized matrix
            weighted_matrix = norm_matrix * weights

            # Step 3: Identify ideal and anti-ideal
            ideal = np.max(weighted_matrix, axis=0) * types + np.min(weighted_matrix, axis=0) * (1 - types)
            anti_ideal = np.min(weighted_matrix, axis=0) * types + np.max(weighted_matrix, axis=0) * (1 - types)

            # Step 4: Calculate distances
            dist_ideal = np.sqrt(((weighted_matrix - ideal) ** 2).sum(axis=1))
            dist_anti = np.sqrt(((weighted_matrix - anti_ideal) ** 2).sum(axis=1))

            # Step 5: Calculate scores
            scores = dist_anti / (dist_ideal + dist_anti)
            result_df = pd.DataFrame({
                "Alternative": df.index,
                "TOPSIS Score": scores,
                "Rank": scores.argsort()[::-1].argsort() + 1
            }).sort_values(by="TOPSIS Score", ascending=False)

            st.write("### TOPSIS Results")
            st.dataframe(result_df.set_index("Alternative"))

        except Exception as e:
            st.error(f"Error in processing: {e}")
else:
    st.info("Awaiting CSV upload.")

st.markdown("---")
st.caption("© 2025 DecideX | Powered by Streamlit & GitHub | MCDM with TOPSIS")
