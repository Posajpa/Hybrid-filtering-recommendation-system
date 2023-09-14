import random
import time
import logging
import math
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


def weighted_sum(similar_sets, user_settings_list, q_mean):
    under_sum = []
    over_sum = []
    lll = [v for v in similar_sets.values() if v > 0]
    summ = sum(lll) - 1
    lenn= len(lll) - 1

    similar_mean = summ / lenn if lenn != 0 else 0
    q_m = q_mean if str(q_mean).lower() != "nan" else 0


    for i, j in zip(user_settings_list, similar_sets.values()):
        if j >= similar_mean and str(i).lower() != "nan":
            over_sum.append((i - q_m) * j)
            under_sum.append(j)
    if sum(under_sum) == 0 or sum(over_sum) == 0:
        return 0

    s = sum(over_sum)
    u = sum(under_sum)
    p = (s / u) + q_m
    return p
    # return 0


def jaccard_similarity(A, B):
    nominator = A.intersection(B)
    denominator = A.union(B)
    similarity = len(nominator) / len(denominator)
    return similarity


def extract_attributes(query_set):
    query_set_ext = {}
    for key_in, value_in in query_set.items():
        k_list = []
        for val in value_in:
            attr_name = val.split("=")[0]
            attr_values = val.split("=")[1]
            for i in attr_values.split(","):
                k_list.append(attr_name + "=" + i.strip())
        query_set_ext[key_in] = k_list
    return query_set_ext


def nested_jackard_sim_of_items(query_set):
    similar_items = {}
    for key, value in query_set.items():
        kk = {}
        for key_in, value_in in query_set.items():
            kk[key_in] = jaccard_similarity(set(value), set(value_in))
        similar_items[key] = kk
    return similar_items


def cluster_queries(df, k_number):
    kmeans = KMeans(n_clusters=k_number).fit(df)
    labels = kmeans.labels_
    df["labels"] = labels
    return df


def elbow_analysis(df):
    inertia = []
    K = range(1, 10)
    for k in K:
        print(k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()
    return plt


def read_query_set(path):
    query_set = {}
    with open(path) as f:
        lines = f.read().splitlines()
        for line in lines:
            line_l = line.split(";")
            query_set[line_l[0]] = line_l[1:]
    return query_set


def extract_to_attr_df(query_set):
    rows = set()
    cols = set()
    for key, values in query_set.items():
        cols.add(key)
        internal_rows = []
        for val in values:
            attr_name = val.split("=")[0]
            attr_values = val.split("=")[1]
            for i in attr_values.split(","):
                rows.add(attr_name + "=" + i.strip())
                internal_rows.append(attr_name + "=" + i.strip())
        query_set[key] = internal_rows

    df = pd.DataFrame(0, index=list(sorted(cols)), columns=list(sorted(rows)))
    for key, cols in query_set.items():
        for col in cols:
            df.at[key, col] = 1
    return df


def find_top_k(utility_matrix_blank_filled, df_clustered, top_k, path):
    top_k_for_user = {}
    for user in utility_matrix_blank_filled.index:
        if str(list(utility_matrix_blank_filled.loc[user].sort_values(ascending=False, na_position="last").head(1))[
                   0]).lower() == "nan" or \
                list(utility_matrix_blank_filled.loc[user].sort_values(ascending=False, na_position="last").head(1))[
                    0] == 0:
            ### need to function for cold start
            print(
                list(df_clustered.sort_values(['mean'], ascending=False).groupby('labels').head(top_k).index)[0:top_k])
            top_k_for_user[user] = list(
                df_clustered.sort_values(['mean'], ascending=False).groupby('labels').head(top_k).index)[0:top_k]
        else:
            l1 = list(utility_matrix_blank_filled.loc[user].sort_values(ascending=False).head(top_k - 2).index)
            l2 = list(df_clustered.sort_values(['mean'], ascending=False).groupby('labels').head(2).index)[0:2]
            top_k_for_user[user] = l1 + l2
    top_k_df = pd.DataFrame(top_k_for_user)
    top_k_df.to_csv(path, index=True)


def evaluation(similar_sets, utility_matrix, query_means):
    utility_matrix_pred = utility_matrix.copy()
    test_set = []
    pred_set = []
    for user, row in utility_matrix_pred.iterrows():  # iterate over rows
        for query, rating in row.items():
            if str(rating).lower() != "nan" and random.randint(0, 3) == 0:
                utility_matrix_pred.at[user, query] = -1

    for user, row in utility_matrix_pred.iterrows():  # iterate over rows
        for query, rating in row.items():
            if rating == -1:
                print(user, query)
                w = weighted_sum(similar_sets[query], utility_matrix.loc[user].tolist(), query_means[query])
                pred_set.append(round(w))
                test_set.append(utility_matrix[query][user])
    print(test_set)
    print(pred_set)
    MAE = mae(test_set, pred_set)

    MSE = np.square(np.subtract(test_set, pred_set)).mean()
    RSME = math.sqrt(MSE)
    return [MAE, RSME]


if __name__ == '__main__':

    logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    start = time.time()

    query_set = read_query_set("data/query_set.csv")

    utility_matrix = pd.read_csv("data/utility_matrix.csv", sep=",", index_col=0)
    utility_matrix_blank_filled = pd.DataFrame(data=None, columns=utility_matrix.columns, index=utility_matrix.index)
    to_fill = utility_matrix.copy()

    query_means = utility_matrix.mean(axis=0)
    query_set_ext = extract_attributes(query_set)
    similar_sets = nested_jackard_sim_of_items(query_set_ext)

    ### iterate through all cells
    for user_to, row in utility_matrix.iterrows():  # iterate over rows
        print(user_to)
        for query_to, rating in row.items():
            if str(rating).lower() == "nan":
                a = similar_sets[query_to]
                b = utility_matrix.loc[user_to]
                c = query_means[query_to]
                prediction = weighted_sum(a, b, c)
                # prediction = weighted_sum(similar_sets[query_to], utility_matrix.loc[user_to], query_means[query_to])
                utility_matrix_blank_filled.at[user_to, query_to] = round(prediction)
                to_fill.at[user_to, query_to] = round(prediction)
    to_fill.round().to_csv('data/out_cb.csv', index=True)
    duration = time.time() - start

    ### ELBOW METHOD FOR K MEANS -> choose k

    # print("elbow2")
    # query_dict = read_query_set("data/query_set.csv")
    # df = extract_to_attr_df(query_dict)

    a = True  # a = true -> do clustering, top-k, evaluation
    if a == True:
        ### CLUSTERING
        print("clustering process")
        query_dict = read_query_set("data/query_set.csv")
        df = extract_to_attr_df(query_dict)
        df_clustered = cluster_queries(df, 5)
        df_clustered["mean"] = utility_matrix.mean(axis=0)

        ### TOP-K
        print("top-k process")
        find_top_k(utility_matrix_blank_filled, df_clustered, 20, "data/out_tk.csv")

        ### EVALUATION
        print("evaluation process")
        eva_res = evaluation(similar_sets, utility_matrix, query_means)

        ### LOG RESULT
        print("time result = ", duration,
              " -  MAE = " + str(round(eva_res[0], 2)) + " - RMSE = " + str(round(eva_res[1], 2)))
        logging.warning("time result = " + str(round(duration, 5)) + " - " + "matrix size = " + str(
            utility_matrix.shape) + " -  MAE = " + str(round(eva_res[0], 2)) + " - RMSE = " + str(
            round(eva_res[1], 2)) + "--evaluation")
    else:
        print("time result = ", duration)
        logging.warning("time result = " + str(round(duration, 5)) + " - " + "matrix size = " + str(
            utility_matrix.shape))
