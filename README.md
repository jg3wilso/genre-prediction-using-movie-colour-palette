# genre-prediction-using-movie-colour-palette

#Overview
The colour palette of movies belonging to different genres are rather unique. For eg. horror movies have a darker tone. This project 
uses machine learning to predict the genre of the movie based on its colour palette.

#Methodology
A collection of 1000 movies and their related colour palettes were obtained. The python script was then used to predict the genre of the
movie based on the colour palette. 5 main genres were considered for this: action, comedy, horror, drama, and family.

# Python Script
The python script takes an input file of movies and their corresponding colour palette. Using multinomial logistic regression, a model is
trained using Stratified K Folds. As output, the genre of the movies are predicted using the model which is then cross-validated to 
evaluate the effectiveness of the model.

#Results
The model could predict action movies quite accurately with a prediction accuracy of 68%. The overall model had an accuracy of 37%


