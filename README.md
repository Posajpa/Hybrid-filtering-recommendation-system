# Hybrid-filtering-recommendation-system
######## About project

data_mining.zip file contains project file works on Pycharm. 

### Data preparation code
	data_prepare_db_movie_film.py

### Recommendation system code
	mining.py

### Results - Data folder contains all data (input, output)

#input files
	filmtv_movies.csv - main database
	query_set.csv - query set, separated by ";"
	user_id.csv - user ids
	utility_matrix.csv - utility matrix with sparsity 

#output files
	out_cb.csv - main result, contains utility matrix with predicted values
	out_tk.csv - top 20 queries recommended for each user


#Log file - contains runtime duration, MAE, RMSQ, ... etc
	app.log
