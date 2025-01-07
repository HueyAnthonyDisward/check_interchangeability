import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder


def calculate_interchangeability_score(dataset1_path, dataset2_path):
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)

    #Độ tương tự cua cột
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    column_similarity = len(columns1.intersection(columns2)) / len(columns1.union(columns2))
    print(f"Column Similarity: {column_similarity:.2f}")

    #Độ tương tự của hàng
    common_rows = pd.merge(df1, df2, how='inner')
    row_similarity = len(common_rows) / min(len(df1), len(df2))
    print(f"Row Similarity: {row_similarity:.2f}")

    #Xét giá trị
    overlapping_columns = list(columns1.intersection(columns2))
    value_similarities = []

    for col in overlapping_columns:
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            values1 = set(df1[col].dropna())
            values2 = set(df2[col].dropna())
            overlap_ratio = len(values1.intersection(values2)) / len(values1.union(values2))
            value_similarities.append(overlap_ratio)
        else:
            encoder = LabelEncoder()
            try:
                col1 = encoder.fit_transform(df1[col].dropna().astype(str))
                col2 = encoder.fit_transform(df2[col].dropna().astype(str))
                similarity = jaccard_score(col1, col2, average='micro')
                value_similarities.append(similarity)
            except:
                continue

    if value_similarities:
        value_similarity = sum(value_similarities) / len(value_similarities)
    else:
        value_similarity = 0
    print(f"Value Similarity: {value_similarity:.2f}")

    #Chấm điểm
    interchangeability_score = (column_similarity + row_similarity + value_similarity) / 3
    print(f"Interchangeability Score: {interchangeability_score:.2f}")

    return interchangeability_score



if __name__ == "__main__":
    dataset1 = "test.csv"
    dataset2 = "test.csv"

    score = calculate_interchangeability_score(dataset1, dataset2)
    print(f"Final Interchangeability Score: {score:.2f}")
