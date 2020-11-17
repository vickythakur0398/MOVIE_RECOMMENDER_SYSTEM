import pandas as pd
import numpy as np

# this is use for calculating the cosine value by using skikit 
# library so if we have set of data it will simply check from that data and will find
# find out how many times it is repeating this is done via countVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
# here i have used cosine_similarity to calculate the similarity among the repeated 
# values so by matching these values we can figure out how close the movies are
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File 
df = pd.read_csv("movie_dataset.csv")

#print df.columns
# df.head()
# print(df)
# df.columns()
# printf df.columns 




##Step 2: Select Features
# this feature is selected by the coloumn and these are the one who have to go through the
# countVectorsize and cosine_similarity so its will computate the values based on that we will see
# which are more relatable 
features = ['keywords' , 'cast' , 'genres' , 'director']


##Step 3: Create a column in DF which combines all selected features
# here i am combing all the features it will take all the features and combine it into one string and based upon that
# wea are returning the big string which is combination of all the column


# this i have used here because i was getting error that few of the keywords which i have selected
# conatains Nan which is causing error here so we have converted all of them into empty string
for feature in features:
	df[feature] = df[feature].fillna('')


	# here i have combined all the data of columnn into big string
def combine_features(row):
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	

	# now we have to call that i.e we have to apply that funcn so that it get applied to all of its dataframe
	# we need to use the apply method
df["combined_features"] = df.apply(combine_features,axis=1)

# now we already have a similarity matrix to find the movie similar in the descending order
# s1 is to get into the row of the matrix and we are going to enumerate on the list so the list is converted into list of tupels
# now based on the tuples we have to sort this and whatever is most near are considered to be good and we need index of that movie
 # that movie index indx its is done via helping function
 # and we want this in descending order beacsue we want the similar movie together




##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 

movie_user_likes = "Gone Girl"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)


## Step 8: Print titles of first 50 movies

#we need to get list of those for first 50 and just print those 50 thats good work done

i=0
print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print("MOVIE SIMILAR TO - >  " + a + " IS ->>>>  "+ get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break
