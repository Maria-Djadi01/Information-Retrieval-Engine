from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
import glob
import re
import pandas as pd
import math
import nltk


# load documents
def load_documents(path):
    documents = glob.glob(path)
    return documents


# documents reader
def documents_reader(documents_paths):
    corpus = []
    for doc in documents_paths:
        with open(doc, "r") as file:
            corpus.append(file.read().replace("\n", ""))
    return corpus


# documents preprocessing
def tokenize_doc(document, regex):
    document = document.lower()
    if regex:
        reg = nltk.RegexpTokenizer(
            r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*"
        )
        doc = reg.tokenize(document)
    else:
        doc = document.split()
    return doc


def porter_stemmer_doc(document):
    stemmer = PorterStemmer()
    doc_stem = [stemmer.stem(word) for word in document]
    return doc_stem


def lancaster_stemmer_doc(document):
    stemmer = LancasterStemmer()
    doc_stem = [stemmer.stem(word) for word in document]
    return doc_stem


def preprocess_doc(document, regex, porter_stemmer):
    if regex:
        document = tokenize_doc(document, True)
    else:
        document = tokenize_doc(document, False)

    if porter_stemmer:
        document = porter_stemmer_doc(document)
    else:
        document = lancaster_stemmer_doc(document)

    # remove the stop words
    stop_words = set(stopwords.words("english"))
    document = [word for word in document if word not in stop_words]
    return document


def caculate_num_docs_term(term, freq_index):
    num_docs_term = 0
    for doc in freq_index.keys():
        if term in freq_index[doc]:
            num_docs_term += 1
    return num_docs_term


def build_freq_index(documents_folder_path, regex, porter_stemmer):
    loaded_documents = load_documents(documents_folder_path)
    print(loaded_documents)
    documents = documents_reader(loaded_documents)
    documents_names = [doc.split("\\")[1].split(".")[0] for doc in loaded_documents]

    # preprocess the documents
    corpus = []
    for doc in documents:
        corpus.append(preprocess_doc(doc, regex, porter_stemmer))

    freq_index = {}
    for i, doc in enumerate(corpus):
        doc_freq = {}
        for term in doc:
            if term in doc_freq:
                doc_freq[term] += 1
            else:
                doc_freq[term] = 1
        freq_index[documents_names[i]] = doc_freq

    # convert frequency index to dataframe

    freq_index_df = {"Term": [], "Document": [], "Frequency": [], "Weight": []}

    num_docs = len(freq_index.keys())
    for doc in freq_index.keys():
        max_doc_freq = max(freq_index[doc].values())
        for term in freq_index[doc]:
            # number of documents where the term is mentioned
            num_docs_term = caculate_num_docs_term(term, freq_index)
            freq_index_df["Term"].append(term)
            freq_index_df["Document"].append(doc)
            freq_index_df["Frequency"].append(freq_index[doc][term])
            freq_index_df["Weight"].append(
                # round(
                #     freq_index[doc][term]
                #     / max_doc_freq
                #     * math.log10((num_docs / num_docs_term) + 1),
                #     4,
                # )
                freq_index[doc][term]
                / max_doc_freq
                * math.log10((num_docs / num_docs_term) + 1),
            )

    freq_index_df = pd.DataFrame(freq_index_df)
    # save it intp csv file
    freq_index_df.to_csv(
        f"data/frequency_indexes/frequency_index_pre_{'porter' if porter_stemmer else 'lancaster'}_{'regex' if regex else 'split'}.csv",
        index=False,
    )


def query_find(query, regex, porter_stemmer):
    if regex and porter_stemmer:
        query_pr = preprocess_doc(query, regex=True, porter_stemmer=True)
        df = pd.read_csv("data/frequency_indexes/frequency_index_pre_porter_regex.csv")
    elif regex and not porter_stemmer:
        query_pr = preprocess_doc(query, regex=True, porter_stemmer=False)
        df = pd.read_csv(
            "data/frequency_indexes/frequency_index_pre_lancaster_regex.csv"
        )
    elif not regex and porter_stemmer:
        query_pr = preprocess_doc(query, regex=False, porter_stemmer=True)
        df = pd.read_csv("data/frequency_indexes/frequency_index_pre_porter_split.csv")
    else:
        query_pr = preprocess_doc(query, regex=False, porter_stemmer=False)
        df = pd.read_csv(
            "data/frequency_indexes/frequency_index_pre_lancaster_split.csv"
        )
    return df[df["Term"].isin(query_pr)], df, query_pr


def search_document(documents_number, regex, porter_stemmer):
    if regex and porter_stemmer:
        df = pd.read_csv("data/frequency_indexes/frequency_index_pre_porter_regex.csv")
    elif regex and not porter_stemmer:
        df = pd.read_csv(
            "data/frequency_indexes/frequency_index_pre_lancaster_regex.csv"
        )
    elif not regex and porter_stemmer:
        df = pd.read_csv("data/frequency_indexes/frequency_index_pre_porter_split.csv")
    else:
        df = pd.read_csv(
            "data/frequency_indexes/frequency_index_pre_lancaster_split.csv"
        )
    return df[df["Document"] == documents_number]


def scalar_product(query, regex, porter_stemmer):
    _, df, query_precessed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    documents = df["Document"].unique()
    df_sum = {
        "Document": [],
        "Relevence": [],
    }
    for doc in documents:
        relvence = 0
        for term in query_precessed:
            term_list = df[df["Document"] == doc]["Term"].tolist()
            if term in term_list:
                weight = df[(df["Document"] == doc) & (df["Term"] == term)][
                    "Weight"
                ].values[0]
                relvence += weight
        if relvence != 0:
            df_sum["Document"].append(doc)
            df_sum["Relevence"].append(relvence)
    return pd.DataFrame(df_sum).sort_values(by=["Relevence"], ascending=False)


def cosin_similarity(query, regex, porter_stemmer):
    _, df, query_processed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    documents = df["Document"].unique()

    df_sum = pd.DataFrame(columns=["Document", "Relevance"])

    for doc in documents:
        df_doc = df[df["Document"] == doc]
        df_doc["Weight"] = df_doc["Weight"].apply(lambda x: x**2)
        w_sum_squared = df_doc["Weight"].sum()
        v_sum_squared = len(query_processed)

        relevance = 0
        for term in query_processed:
            term_rows = df[(df["Document"] == doc) & (df["Term"] == term)]
            if not term_rows.empty:
                weight = term_rows["Weight"].values[0]
                relevance += weight

        v_sum_sqrt = math.sqrt(v_sum_squared)
        w_sum_sqrt = math.sqrt(w_sum_squared)

        if v_sum_sqrt != 0 and w_sum_sqrt != 0:
            relevance = relevance / (v_sum_sqrt * w_sum_sqrt)
        if relevance != 0:
            df_sum = df_sum.append(
                {"Document": doc, "Relevance": relevance}, ignore_index=True
            )

    df_sum = df_sum.sort_values(by=["Relevance"], ascending=False, ignore_index=True)

    return df_sum


def jaccard_measure(query, regex, porter_stemmer):
    _, df, query_processed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    documents = df["Document"].unique()

    df_sum = pd.DataFrame(columns=["Document", "Relevance"])

    for doc in documents:
        df_doc = df[df["Document"] == doc]
        df_doc["Weight"] = df_doc["Weight"].apply(lambda x: x**2)
        w_sum_squared = df_doc["Weight"].sum()
        v_sum_squared = len(query_processed)
        relevance = 0
        for term in query_processed:
            term_rows = df[(df["Document"] == doc) & (df["Term"] == term)]
            if not term_rows.empty:
                weight = term_rows["Weight"].values[0]
                relevance += weight
                # v_sum_squared += 1

        if relevance != 0:
            relevance = relevance / (v_sum_squared + w_sum_squared - relevance)
            df_sum = df_sum.append(
                {"Document": doc, "Relevance": relevance}, ignore_index=True
            )

    df_sum = df_sum.sort_values(by=["Relevance"], ascending=False, ignore_index=True)
    return df_sum


def model_BM25(query, regex, porter_stemmer, k, b):
    _, df, query_processed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    print(_, query_processed)
    df_sum = {"Document": [], "Relevence": []}

    documents = df["Document"].unique()
    # Calculate the mean number of terms in documents
    doc_term_counts = []
    for doc in documents:
        term_count = df[df["Document"] == doc]["Term"].nunique()
        doc_term_counts.append(term_count)

    avdl = sum(doc_term_counts) / len(doc_term_counts)

    # Number of ducuments
    N = len(doc_term_counts)
    for doc in documents:
        term_sum = 0
        for term in query_processed:
            term_list = df[df["Document"] == doc]["Term"].tolist()
            if term in term_list:
                freq = df[(df["Document"] == doc) & (df["Term"] == term)][
                    "Frequency"
                ].values[0]
                dl = df[df["Document"] == doc]["Term"].nunique()
                ni = df[df["Term"] == term]["Document"].nunique()
                term_sum += (
                    freq / (k * (1 - b) + b * (dl / avdl) + freq)
                ) * math.log10((N - ni + 0.5) / (ni + 0.5))
        if term_sum != 0:
            df_sum["Document"].append(doc)
            df_sum["Relevence"].append(term_sum)
    return pd.DataFrame(df_sum).sort_values(by=["Relevence"], ascending=False)


def mod_BM25(query, regex, porter_stemmer, k, b):
    _, df, query_processed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    df_sum = pd.DataFrame(columns=["Document", "Relevance"])

    documents = df["Document"].unique()
    N = len(documents)
    avdl = (df["Frequency"].sum()) / N
    term_in_doc = _.groupby("Term", as_index=False)["Document"].count()
    for doc in documents:
        dl = df[df["Document"] == doc]["Frequency"].sum()
        relevance = 0
        for term in query_processed:
            ni = term_in_doc[term_in_doc["Term"] == term]["Document"]
            doc_df = df[df["Document"] == doc]
            if term in doc_df["Term"].values:
                freq = doc_df[doc_df["Term"] == term]["Frequency"].values[0]
                right = freq / ((k * ((1 - b) + b * (dl / avdl))) + freq)
                left = math.log10((N - ni + 0.5) / (ni + 0.5))
                relevance += right * left
        if relevance != 0:
            df_sum = df_sum.append(
                {"Document": doc, "Relevance": relevance}, ignore_index=True
            )
    df_sum = df_sum.sort_values(by=["Relevance"], ascending=False, ignore_index=True)
    return df_sum


mod_BM25("Documents AND NOT ranking OR queries OR GPT-3.5", True, True, 2, 0.75)


# check if the query is valid or not
def valid(query_terms):
    if query_terms[0] in ["AND", "OR"] or query_terms[-1] in ["AND", "OR", "NOT"]:
        return False
    for i in range(len(query_terms) - 1):
        if query_terms[i] in ["AND", "OR"] and query_terms[i + 1] in ["AND", "OR"]:
            return False
        if query_terms[i] == "NOT" and query_terms[i + 1] in ["AND", "OR", "NOT"]:
            return False
        if query_terms[i] not in ["AND", "OR", "NOT"] and query_terms[i + 1] not in [
            "AND",
            "OR",
            "NOT",
        ]:
            return False
    return True


def boolean_similarity(doc_terms, query):
    query_terms = query.split()
    for i in range(len(query_terms)):
        if query_terms[i] not in ["AND", "OR", "NOT"]:
            query_terms[i] = " ".join(preprocess_doc(query_terms[i], True, True))

    for i in range(len(query_terms)):
        if query_terms[i] not in ["AND", "OR", "NOT"]:
            if query_terms[i] in doc_terms:
                query_terms[i] = 1
            else:
                query_terms[i] = 0

    # remove NOT
    i = 0
    while i < len(query_terms):
        if query_terms[i] == "NOT":
            if query_terms[i + 1] == 0:
                query_terms[i + 1] = 1
            else:
                query_terms[i + 1] = 0
            query_terms.pop(i)
        else:
            i += 1

    relevant = 0
    i = 0
    while i < len(query_terms):
        if query_terms[i] == "AND":
            relevant = query_terms[i - 1] and query_terms[i + 1]
            query_terms[i + 1] = relevant
            query_terms.pop(i)
            i += 1
        elif query_terms[i] == "OR":
            relevant = query_terms[i - 1] or query_terms[i + 1]
            query_terms[i + 1] = relevant
            query_terms.pop(i)
            i += 1
        else:
            i += 1

    if relevant:
        relevant = "YES"
    else:
        relevant = "NO"
    return relevant


def boolean_model(query):
    df_result = pd.DataFrame(columns=["Document", "Relevance"])
    if query == "":
        return df_result
    else:
        query_terms = query.split()
        if valid(query_terms):
            df = pd.read_csv(
                "data/frequency_indexes/frequency_index_pre_porter_regex.csv"
            )
            documents = df["Document"].unique()
            doc_terms = df[df["Document"] == "D1"]["Term"].to_list()
            for doc in documents:
                doc_terms = df[df["Document"] == doc]["Term"].to_list()
                relevance = boolean_similarity(doc_terms, query)
                if relevance == "YES":
                    df_result = df_result.append(
                        {"Document": doc, "Relevance": relevance}, ignore_index=True
                    )
    return df_result


def precision(docs_retrieved, denominator):
    nb_docs_true = 0
    for doc in docs_retrieved:
        if doc:
            nb_docs_true += 1

    return nb_docs_true / denominator


def recall(docs_retrieved, nb_true_documents):
    nb_docs_true = 0
    for doc in docs_retrieved:
        if doc:
            nb_docs_true += 1
    return nb_docs_true / nb_true_documents


def f_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def evaluation_metrics(regex, porter_stemmer, nb_query):
    query_df = pd.read_csv("data/test/queries.csv")
    query = query_df.iloc[nb_query - 1]["query"]

    judgement_df = pd.read_csv("data/test/judgements.csv").iloc[:, :-1]
    judgement_df = judgement_df[judgement_df["query_number"] == nb_query]

    df, _, query_processed = query_find(
        query, regex=regex, porter_stemmer=porter_stemmer
    )
    docs_true = judgement_df["document"].tolist()
    docs_retrieved = df["Document"].unique().tolist()

    retrieved_documents = []
    for doc in docs_retrieved:
        retrieved_documents.append(doc in docs_true)

    prec = precision(retrieved_documents, len(retrieved_documents))
    prec_5 = precision(retrieved_documents[:5], 5)
    prec_10 = precision(retrieved_documents[:10], 10)
    rec = recall(retrieved_documents, len(docs_true))
    f_sc = f_score(prec, rec)

    return query, prec, prec_5, prec_10, rec, f_sc


evaluation_metrics(True, True, 2)


# build_freq_index("data/documents/*.txt", True, True)
# build_freq_index("data/documents/*.txt", True, False)
# build_freq_index("data/documents/*.txt", False, True)
# build_freq_index("data/documents/*.txt", False, False)


# ----------------------------------
# TEST BOOLEAN MODEL
# ---------------------------------

# query = "Documents AND NOT ranking OR queries OR GPT-3.5"

# boolean_model(query)
