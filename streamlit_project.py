# Importing needed libraries

from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
    
# importing library for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

header = st.container()

data = st.container()

visual = st.container()

training_rf_model = st.container()

rf_model_evaluation = st.container()

decision_tree_model = st.container()

decision_tree_evaluation = st.container()

k_means = st.container()


with header:
    st.title('Welcome to my Machine Learning project!')

with data:
    st.header('Today we will do predictions using different classification models and also make clusters from our data using KMeans algorithm!')

    st.subheader('Marketing Campaign dataset consists of the following data:')

    dataset = pd.read_csv('marketing_campaign.csv', sep = '\t')

    #Dropping missing values
    dataset.dropna(axis=0, inplace=True)

    #Making sure that dataset has no duplicates
    dataset.drop_duplicates(keep='first', inplace=True)
    
    st.write(dataset.head(6))


lc_rf = Image.open('learning_curve_for_random_forest_classifier.png')
lc_dt = Image.open('learning_curve_for_decision_tree_classifier.png')

with visual:
    st.subheader('Visualizing data:')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    df1 = dataset[["Year_Birth", "Recency", "MntWines", "Income"]]
    df1.hist(color='green')
    plt.show()
    st.pyplot()

    df2 = dataset[["MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts",]]
    df2.hist( color='brown')
    plt.show()
    st.pyplot()

    df3 = dataset[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]]
    df3.hist( color='blue')
    plt.show()
    st.pyplot()

    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Education", data = dataset)
    plt.title('Education column', fontsize=14)
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Marital_Status", data = dataset)
    plt.title('Marital_status column', fontsize=14)
    st.pyplot(fig2)

    fig3 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Marital_Status", data = dataset)
    plt.title('Marital_status column', fontsize=14)
    st.pyplot(fig3)

    fig3 = plt.figure(figsize=(10, 5))
    sns.countplot(x="Education", hue="Response",data=dataset)
    plt.title('Education and Response', fontsize=14)
    st.pyplot(fig3)

    fig3 = plt.figure(figsize=(10, 5))
    sns.countplot(x="Marital_Status", hue="Response",data=dataset)
    plt.title('Marital_Status and Response', fontsize=14)
    st.pyplot(fig3)



    st.write(dataset.describe())

    outliers = plt.figure()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.boxplot(data=dataset[["Income"]], orient="h")
    st.pyplot(outliers)

    for x in dataset[['Income']]:
        dataset.loc[dataset[x] > 120000,x] = 120000

    st.subheader('Changing the values of outliers in Income column to a max of 120k')
    remove_outliers = '''for x in dataset[['Income']]:\n\tdataset.loc[dataset[x] > 120000,x] = 120000'''
    
    st.code(remove_outliers, language='python')


with training_rf_model:
    st.subheader('Splitting dataset into features and target')
    features = dataset[["Year_Birth", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Complain"]]
    target = dataset["Response"]

    f_t_split_code = '''features = data[["Year_Birth", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Complain"]]\ntarget = data["Response"]'''
    st.code(f_t_split_code, language='python')

    st.subheader('Splitting dataset into testing and training parts:')
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, random_state = 0, test_size=0.20)

    test_train_split_code = '''x_train, x_valid, y_train, y_valid = train_test_split(features, target, random_state = 0, test_size=0.20)'''
    st.code(test_train_split_code, language='python')

    st.subheader('Creation and training of RandomForest model with n_estimators=20')
    r_forest = RandomForestClassifier(n_estimators=20)
    r_forest.fit(x_train, y_train)

    r_f_model = '''r_forest = RandomForestClassifier(n_estimators=20)\nr_forest.fit(x_train, y_train)'''
    st.code(r_f_model, language='python')

with rf_model_evaluation:
    st.subheader('Evaluation of RandomForestClassifier')

    col1, col2 = st.columns(2)


    col1.text('Prediction Accuracy:')
    col1.write(accuracy_score(y_valid, r_forest.predict(x_valid)))

    col1.text('Mean absolute error:')
    col1.write(mean_absolute_error(y_valid, r_forest.predict(x_valid)))

    col1.text('Mean squared error:')
    col1.write(mean_squared_error(y_valid, r_forest.predict(x_valid)))

    col1.text('R squared score:')
    col1.write(r2_score(y_valid, r_forest.predict(x_valid)))

    fig_c_m1 = plt.figure(figsize=(14, 8))
    cm = confusion_matrix(y_valid, r_forest.predict(x_valid))
    ax= plt.subplot()
    col2.markdown('**Confusion matrix:**')
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    col2.pyplot(fig_c_m1)


with decision_tree_model:
    st.subheader('Creation and training of DecisionTree model with random_state=10')
    # creating and training Decision tree model
    dtree_model = DecisionTreeClassifier(random_state=10)
    dtree_model.fit(x_train, y_train)


    d_t_model = '''dtree_model = DecisionTreeClassifier(random_state=10)\ndtree_model.fit(x_train, y_train)'''
    st.code(d_t_model, language='python')

with decision_tree_evaluation:
    st.subheader('Evaluation of DecisionTreeClassifier')

    col1, col2 = st.columns(2)


    col1.text('Prediction Accuracy:')
    col1.write(accuracy_score(y_valid, dtree_model.predict(x_valid)))


    col1.text('Mean absolute error:')
    col1.write(mean_absolute_error(y_valid, dtree_model.predict(x_valid)))

    col1.text('Mean squared error:')
    col1.write(mean_squared_error(y_valid, dtree_model.predict(x_valid)))

    col1.text('R squared score:')
    col1.write(r2_score(y_valid, dtree_model.predict(x_valid)))

    fig_c_m2 = plt.figure(figsize=(14, 8))
    cm = confusion_matrix(y_valid, dtree_model.predict(x_valid))
    ax= plt.subplot()
    col2.markdown('**Confusion matrix:**')
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    col2.pyplot(fig_c_m2)

    st.markdown('**Learning curve for RandomForestClassifier:**')
    st.image(lc_rf, use_column_width=True)

    st.markdown('**Learning curve for DecisionTreeClassifier:**')
    st.image(lc_dt, use_column_width=True)

with k_means:

    st.subheader('Let\' create KMeans clustering model!')

    wcss1=[]
    X1= dataset.iloc[:, [4,28]].values

    for i in range(1,11):
        kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
        kmeans.fit(X1)
        wcss1.append(kmeans.inertia_)

    st.markdown('**Visualizing the ELBOW method to get the optimal value of clusters**')

    num_of_clusters = plt.figure(figsize=(10, 6))
    plt.plot(range(1,11), wcss1)
    plt.title('The Elbow Method')
    plt.xlabel('number of clusters')
    plt.show()
    st.pyplot(num_of_clusters)

    kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X1)


    col_1, col_2, col_3 = st.columns(3)




    vis_clusters1 = plt.figure(figsize=(10, 6))
    plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('Income')
    plt.ylabel('Response')
    plt.legend()
    plt.show()
    col_1.pyplot(vis_clusters1)


    X2= dataset.iloc[:, [8,28]].values

    kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X2)


    vis_clusters2 = plt.figure(figsize=(10, 6))
    plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('Recency')
    plt.ylabel('Response')
    plt.legend()
    plt.show()
    col_2.pyplot(vis_clusters2)

    wcss3=[]

    X3= dataset.iloc[:, [9,28]].values

    kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X2)


    vis_clusters3 = plt.figure(figsize=(10, 6))
    plt.scatter(X3[y_kmeans == 0, 0], X3[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X3[y_kmeans == 1, 0], X3[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X3[y_kmeans == 2, 0], X3[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('MntWines')
    plt.ylabel('Response')
    plt.legend()
    plt.show()
    col_3.pyplot(vis_clusters3)

    X4= dataset.iloc[:, [10,11,12,13,14,15,16,17,18,19]].values

    kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X2)


    vis_clusters4 = plt.figure(figsize=(10, 6))
    plt.scatter(X4[y_kmeans == 0, 0], X4[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X4[y_kmeans == 1, 0], X4[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X4[y_kmeans == 2, 0], X4[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('All columns [10-19]')
    plt.legend()
    plt.show()
    st.pyplot(vis_clusters4)



    # 10,11,12,13,14,15,16,17,18,19