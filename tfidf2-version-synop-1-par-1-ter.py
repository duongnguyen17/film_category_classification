import nltk
import csv

synop = []
genres = []
nb_genre = []

#Lecture des données, regroupement des synopsis de même genre

with open('data.csv', 'r', newline='', encoding='utf-8') as datafile:
    reader = csv.reader(datafile, delimiter = ',')
    i = 0
    for row in reader:
        if row[2] not in genres :
            genres += [row[2]]
            synop += [row[1]]
        else :
            synop[genres.index(row[2])] +=  row[1]
            
    

synop.pop(0)
genres.pop(0)

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
#create an object that will do the lemmatization function
wordnet=WordNetLemmatizer()
#create a new variable that is list to store everything from this paragraph after the lemmatization
corpus = []

#Normalisation dont lemmatisation des synopsis
for i in range(len(synop)):
    review = re.sub('[^a-zA-Z]', ' ', synop[i])
    review_lower = review.lower()
    review_split = review_lower.split()
    review_lem = [wordnet.lemmatize(word) for word in review_split if not word in set(stopwords.words('english'))]
    review_fin = ' '.join(review_lem)
    corpus.append(review_fin)
    

#Création de la matrice lexicon : chaque ligne correspond à un genre, chaque colonne à un mot.
#La valeur Y[i][j] correspond au nombre d'apparitions du mot j dans des synopsis du genre i
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
Y = cv.fit_transform(corpus).toarray() 

Dico = cv.vocabulary_  

#Défintion d'une fonction permettant de sortir l'élément de plys haute valeur d'un dictionnaire

import operator
def get_argmax_key(dictionary):
    return max(dictionary.items(), key=operator.itemgetter(1))[0]

###Calcul du genre le plus probable selon le modèle lexicon

# 1) Calcul du nombre de mots par genre pour le score normalisé
somme = {}
for genre in genres:
    somme[genre]=sum(Y[genres.index(genre)])

meilleur_genre = [] #Contiendra le meilleur genre pour chaque synopsis, modèle non normalisé
meilleur_genre_norm = [] #Contiendra le meilleur genre pour chaque synopsis, modèle normalisé
    
with open('data.csv', 'r', newline='', encoding='utf-8') as datafile:
    reader = csv.reader(datafile, delimiter = ',')
    i = 0
    for row in reader:
        if i in [j for j in range(1,4000)]:
            score_genres = {}
            score_norm = {}
            for genre in genres:
                score_genres[genre]=0
                score_norm[genre]=0
            
            #Normalisation du synopsis testé, dont lemmatisation            
            
            synop_test = row[1]
            review = re.sub('[^a-zA-Z]', ' ', synop_test)
            review_lower = review.lower()
            review_split = review_lower.split()
            review_lem = [wordnet.lemmatize(word) for word in review_split if not word in set(stopwords.words('english'))]
            review_fin = ' '.join(review_lem)
            synop_test_modif = []
            synop_test_modif.append(review_fin)
            
            words_synop = synop_test_modif[0].split(' ')
            
            #Calcul du score de chaque genre pour le synopsis testé            
            
            for genre in genres:
                for w in words_synop:
                    try:
                        score_genres[genre] += Y[genres.index(genre)][Dico[w]]
                        score_norm[genre] += Y[genres.index(genre)][Dico[w]]/somme[genre]
                    except:
                        pass
                    
            #Selection du genre ayant le score le plus haut
            
            meilleur_genre += [get_argmax_key(score_genres)]
            meilleur_genre_norm += [get_argmax_key(score_norm)]
            
            print(i)
        i+=1

#Création de la liste des vrais genres pou comparer aves les genres trouvés

vrai_genre = []
with open('data.csv', 'r', newline='', encoding='utf-8') as datafile:
    reader = csv.reader(datafile, delimiter = ',')
    i = 0
    for row in reader:
        vrai_genre += [row[2]]
vrai_genre.pop(0)

#Compte du nombre de genres corrects

juste = 0
juste_norm = 0
for i in range(len(meilleur_genre)):
    if meilleur_genre[i]==vrai_genre[i]:
        juste += 1
    if meilleur_genre_norm[i]==vrai_genre[i]:
        juste_norm += 1
print(juste)
print(juste_norm)





