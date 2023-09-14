import random
import pandas as pd


def user_ids(path, number_of_users):
    with open(path, 'w') as fp:
        for i in range(number_of_users):
            fp.write("user" + str(i) + "\n")
    print("user id exported")

def query_set(path, path_out, number_of_queries):
    movie_db = pd.read_csv(path)
    actor_sort = query_actor_sort(path)
    column_list = list(movie_db.columns)
    columns_useless = ['filmtv_id', 'title', 'critics_vote', 'public_vote', 'total_votes', 'description', 'notes']

    for c in columns_useless:
        column_list.remove(c)

    print(column_list)
    with open(path_out, 'w') as fp:
        for i in range(0, number_of_queries):
            rand_col_num = random.randint(0, len(column_list))
            chosen_columns = random.sample(column_list, rand_col_num)

            while len(chosen_columns) == 0:
                print(i, rand_col_num)
                rand_col_num = random.randint(0, len(column_list))
                chosen_columns = random.sample(column_list, rand_col_num)
            print(chosen_columns)
            fp.write("Q" + str(i) + ";")
            for col in chosen_columns:
                value = query_set_util(movie_db, col)
                # print(value)
                if col == chosen_columns[-1]:
                    fp.write(col + "=" + query_set_equilizer(col, value, actor_sort) + "\n")
                else:
                    fp.write(col + "=" + query_set_equilizer(col, value, actor_sort) + ";")
    return


def query_set_equilizer(col, value, actor_sort):

    if col in ["actors"]:
        ll = [i.strip() for i in str(value).split(',')]
        if len(ll) > 5:
            d_list = []
            for l in ll:
                d_list.append([l, actor_sort[l]])
            top5act = [i for i, j in sorted(d_list, reverse=True, key=lambda kv: (kv[1], kv[0]))[0:5]]
            return ','.join(top5act)
    return str(value)


def query_set_util(df, col):  # returns random columns with value
    l = df.loc[:, col]
    return random.choice(l)


def query_actor_sort(path):
    movie_db = pd.read_csv(path)
    actors_dict = {}
    for ll in list(movie_db["actors"]):
        if str(ll) == "nan":
            actors_dict["nan"] = 0
            continue
        for l in ll.split(","):
            actors_dict[l.strip()] = 0

    for ll in list(movie_db["actors"]):
        if str(ll) == "nan":
            actors_dict["nan"] = actors_dict["nan"] + 1
            continue
        for l in ll.split(","):
            actors_dict[l.strip()] = actors_dict[l.strip()] + 1

    print(actors_dict)

    return actors_dict


def utility_matrix(path_user, path_query, path_utility_matrix):

    query_set = []
    user_set = []
    with open(path_query) as f:
        lines = f.read().splitlines()
        for line in lines:
            line_l = line.split(";")
            query_set.append((line_l[0]))

    with open(path_user) as f:
        lines = f.read().splitlines()
        for line in lines:
            user_set.append(line)

    with open(path_utility_matrix, 'w') as fp:
        ### WRITE COLS###
        fp.write("" + ",")
        for query in query_set:
            if query == query_set[-1]:
                fp.write(query + "\n")
            else:
                fp.write(query + ",")

        ### WRITE ROWS ###
        for user in user_set:
            fp.write(user + ",")
            for query in query_set:
                rn = random.randint(0, 101)
                rn_yn = bool(random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))  # -> 70 percent non blank cells
                if query == query_set[-1]:
                    if rn_yn:
                        fp.write("" + "\n")
                    else:
                        fp.write(str(rn) + "\n")
                else:
                    if not rn_yn:
                        fp.write("" + ",")
                    else:
                        fp.write(str(random.randint(0, 100)) + ",")
    return


def check_UM_blank_percentage(path):
    matrix = []
    with open(path) as f:
        lines = f.read().splitlines()
        for line in lines:
            matrix.append(line.split(","))

    k = 0
    s = 0
    for l in matrix:
        k = k + l.count('')
        s = s + len(l)
    s = s - len(matrix) - len(matrix[0]) + 1
    return k * 100 / s


if __name__ == '__main__':
    user_ids("data/user_id.csv", 1000)
    query_set("data/filmtv_movies.csv", "data/query_set.csv", 1000)
    utility_matrix("data/user_id.csv", "data/query_set.csv", "data/utility_matrix.csv")
    percent = check_UM_blank_percentage("data/utility_matrix.csv")
    print("percent =", percent)