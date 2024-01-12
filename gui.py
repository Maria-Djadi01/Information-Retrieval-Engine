import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

test_queries = pd.read_csv("data/test/queries.csv")

st.set_page_config(layout="wide")

st.title("Information Retrieval")
section1, section2 = st.columns([3, 2])

with section2:
    matching = st.checkbox("Matching")
    matching_model = st.radio(
        label="Matching",
        options=[
            "Vector Space Model",
            "Probabilistic Model",
            "Boolean Model",
            "Data Mining Model",
        ],
    )
    if matching_model == "Probabilistic Model":
        k = st.text_input(label="K", value=1.5)
        b = st.text_input(label="B", value=0.75)
    if matching_model == "Vector Space Model":
        vec_sec1, vec_sec2 = st.columns([6, 2])
        with vec_sec1:
            vec_mode = st.selectbox(
                label="Vector Mode",
                options=["Scalar Product", "Cosine Similarity", "Jaccard Similarity"],
            )
        with vec_sec2:
            nb_query = st.number_input(label="Number of query", min_value=1, max_value=len(test_queries), value=1, step=1)
    if matching and matching_model == "Vector Space Model":
        # make a random courbe
        x = np.linspace(0, 10, 100)
        y = np.random.randn(100)
        df_chart = pd.DataFrame({"x": x, "y": y})
        st.line_chart(df_chart, x="x", y="y")

with section1:
    check_col1, radio_col2 = st.columns(2)
    # check box
    with check_col1:
        # label of the col
        st.text("Preprocessing")
        check_col1_1, check_col1_2 = check_col1.columns(2)
        regex = check_col1_1.checkbox("Tokenization", value=True)
        porter_stemmer = check_col1_2.checkbox("Porter Stemmer", value=True)

    index = radio_col2.radio(
        label="Index",
        options=["DOCS per TERM", "TERMS per DOC"],
        horizontal=True,
        disabled=matching,
    )
    
    if nb_query:
        query_eval, prec, prec_5, prec_10, rec, f_sc = evaluation_metrics(
                    regex=regex, porter_stemmer=porter_stemmer, nb_query=int(nb_query)
                )
    else:
        query_eval = ""
    query = st.text_input(label="Search", placeholder="Enter your query here", value=query_eval if nb_query is not None else None)

    if index == "DOCS per TERM" and not matching:
        if query:
            query_df, _, _ = query_find(
                query, regex=regex, porter_stemmer=porter_stemmer
            )
            st.dataframe(query_df)
    elif index == "TERMS per DOC" and not matching:
        loaded_documents = load_documents("data/documents/*.txt")
        documents_names = [doc.split("\\")[1].split(".")[0] for doc in loaded_documents]
        # dropdown list of documents
        query = st.selectbox("Select Document", documents_names)
        st.dataframe(search_document(query, regex=regex, porter_stemmer=porter_stemmer))
    elif matching and matching_model == "Vector Space Model":
        if vec_mode == "Scalar Product":
            st.dataframe(
                scalar_product(query, regex=regex, porter_stemmer=porter_stemmer)
            )
            
            sec_prec, sec_prec_5, sec_prec_10, sec_rec, sec_f_sc = st.columns(5)
            with sec_prec:
                st.text(f"Precision: {prec}")
            with sec_prec_5:
                st.text(f"Precision@5: {prec_5}")
            with sec_prec_10:
                st.text(f"Precision@10: {prec_10}")
            with sec_rec:
                st.text(f"Recall: {rec}")
            with sec_f_sc:
                st.text(f"F-Score: {f_sc}")

        elif vec_mode == "Cosine Similarity":
            st.dataframe(
                cosin_similarity(query, regex=regex, porter_stemmer=porter_stemmer)
            )
        elif vec_mode == "Jaccard Similarity":
            st.dataframe(
                jaccard_measure(query, regex=regex, porter_stemmer=porter_stemmer)
            )
    elif matching and matching_model == "Probabilistic Model":
        st.dataframe(
            model_BM25(
                query,
                regex=regex,
                porter_stemmer=porter_stemmer,
                k=float(k),
                b=float(b),
            )
        )
    elif matching and matching_model == "Boolean Model":
        st.dataframe(boolean_model(query))
    elif matching and matching_model == "Data Mining Model":
        pass
