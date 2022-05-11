# Importing needed libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
    
# importing library for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# Dividing our Page into containers
header = st.container()

data = st.container()

visual = st.container()

training_rf_model = st.container()

rf_model_evaluation = st.container()

decision_tree_model = st.container()

decision_tree_evaluation = st.container()

k_means = st.container()

# Header container
with header:
    # Title of a page
    st.title('Welcome to my Machine Learning project!')

# Data container for working with dataset
with data:
    # Text of the header, our goals for a topic
    st.header('Today we will do predictions using different classification models and also make clusters from our data using KMeans algorithm!')

    st.subheader('Marketing Campaign dataset consists of the following data:')

    # reading dataset and storing it into dataframe (dataset)
    dataset = pd.read_csv('marketing_campaign.csv', sep = '\t')

    # showing first 6 elements of a dataset on a page
    st.write(dataset.head(6))

    # more text
    st.markdown('Too much inconvenient data. So "Year_Birth" was replaced to "Age" by substracting from current year birth year. After we created "Spent_total" column, which is the total spending on various items. After we made Education column a bit easier: undergraduate, graduate and postgraduate.')
    st.markdown('')
    st.markdown('Also we checked for unique values and understood that "Z_CostContact", "Z_Revenue" columns have only a single value. With them we dropped unecessary columns as customer-ID and Dt_customer(Date of customer\'s enrollment with the company).')

    # creating Age column based on the difference of current year and Year_birth of a customers
    dataset["Age"] = 2022-dataset["Year_Birth"]

    # Total spendings on various items (sum of all spendings between columns)
    dataset["Spent_total"] = dataset["MntWines"]+ dataset["MntFruits"]+ dataset["MntMeatProducts"]+ dataset["MntFishProducts"]+ dataset["MntSweetProducts"]+ dataset["MntGoldProds"]

    # making Education column a bit simpler to understand (only Undergraduate, graduate and postgraduate)
    dataset["Education"]=dataset["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    # "Z_CostContact", "Z_Revenue" columns consist of only a single value, so has no contribution to common result
    # dropping unecessary columns
    columns_for_dropping = ["Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    dataset = dataset.drop(columns_for_dropping, axis=1)

    # Dropping missing values
    dataset.dropna(axis=0, inplace=True)

    # Making sure that dataset has no duplicates
    dataset.drop_duplicates(keep='first', inplace=True)
    
# loading images into a variables (lc_rf, lc_dt)
lc_rf = Image.open('learning_curve_for_random_forest_classifier.png')
lc_dt = Image.open('learning_curve_for_decision_tree_classifier.png')

# container for visualization
with visual:
    st.subheader('Visualizing data:')

    # writing this line so warning won't appear
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # making histogramms based on 4 columns
    df1 = dataset[["Age", "Recency", "MntWines", "Income"]]
    df1.hist(color='green')
    plt.show()
    st.pyplot()

    #more text
    st.markdown("As we can see most customers are 40-60 years. Recency(number of days since customer's last purchase) varies equally between 0 and 100 days.")
    st.markdown("Income of most customers are less than 100k, about 500 customers have income more than 100k.")

    # making histogramms based on 4 columns
    df2 = dataset[["MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts",]]
    df2.hist( color='brown')
    plt.show()
    st.pyplot()

    #text
    st.markdown("Above we can see amount spent on different items. People spend on wine the most.")

    # making histogramms based on 4 columns
    df3 = dataset[["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]]
    df3.hist( color='blue')
    plt.show()
    st.pyplot()

    #text
    st.markdown("Above we can type of purchase and how many times customers use a certain type.")

    # making countplot based on Education column
    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Education", data = dataset)
    plt.title('Education column', fontsize=14)
    st.pyplot(fig1)

    #text
    st.markdown("Most of the customers are graduated and postgraduated.")

    # making countplot based on Marital_Status column
    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x = "Marital_Status", data = dataset)
    plt.title('Marital_status column', fontsize=14)
    st.pyplot(fig1)

    #text
    st.markdown("Most of the customers are married or together. Decent amount are single.")

    #countplot in a relation of Education to a Response column
    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x="Education", hue="Response",data=dataset)
    plt.title('Education and Response', fontsize=14)
    st.pyplot(fig1)

    #text
    st.markdown("Comparing Educational background of customers and their responses.")

    #countplot in a relation of Marital_Status to a Response column
    fig1 = plt.figure(figsize=(10, 5))
    sns.countplot(x="Marital_Status", hue="Response",data=dataset)
    plt.title('Marital_Status and Response', fontsize=14)
    st.pyplot(fig1)


    st.markdown("Comparing Marital Status of customers and their responses.")
    st.markdown("")

    st.markdown("Below we can see data description:")
    st.write(dataset.describe())

    st.markdown("By description looking at max or min values and mean values of columns we conluded that Income column has outliers:")

    # visualizing outliers of income column
    outliers = plt.figure()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.boxplot(data=dataset[["Income"]], orient="h")
    st.pyplot(outliers)

    for x in dataset[['Income']]:
        dataset.loc[dataset[x] > 120000,x] = 120000

    st.markdown('Changing the values of outliers in Income column to a max of 120k:')
    remove_outliers = '''for x in dataset[['Income']]:\n\tdataset.loc[dataset[x] > 120000,x] = 120000'''
    
    st.code(remove_outliers, language='python')


with training_rf_model:




    st.subheader('Splitting dataset into features and target')
    features = dataset[["Age", "Spent_total", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Complain"]]
    target = dataset["Response"]

    f_t_split_code = '''features = data[["Age", "Spent_total", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Complain"]]\ntarget = data["Response"]'''
    st.code(f_t_split_code, language='python')

    st.subheader('Splitting dataset into testing and training parts:')
    x_train, x_valid, y_train, y_valid = train_test_split(features, target, random_state = 0, test_size=0.20)

    test_train_split_code = '''x_train, x_valid, y_train, y_valid = train_test_split(features, target, random_state = 0, test_size=0.20)'''
    st.code(test_train_split_code, language='python')

    st.subheader('Creating and training of RandomForest model with n_estimators=20')
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
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  
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

    dataset

    st.markdown("TAge not much affects on the result.")




with k_means:

    st.subheader('Let\' create KMeans clustering model!')

    wcss1=[]
    X1= dataset.iloc[:, [6,24]].values

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

    kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X1)

    dataset["clusters"] = kmeansmodel.fit_predict(dataset[dataset.columns[2:]])

    cl_fig8 = plt.figure(figsize=(10, 5))
    sns.countplot(x='clusters',data=dataset)
    plt.title('Ditribution of clusters', fontsize=14)
    st.pyplot(cl_fig8)

    st.markdown("Distribution of customers between clusters.")

    

    labelencoder = LabelEncoder()
    dataset["Education"] = labelencoder.fit_transform(dataset["Education"])
    dataset["Marital_Status"] = labelencoder.fit_transform(dataset["Marital_Status"])


    vis_clusters1 = plt.figure(figsize=(10, 6))
    plt.scatter(X1[y_kmeans == 0, 0], X1[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X1[y_kmeans == 1, 0], X1[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X1[y_kmeans == 2, 0], X1[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('Wine')
    plt.ylabel('Age')
    plt.legend()
    plt.show()
    st.pyplot(vis_clusters1)

    st.markdown("We can see that age doesnot affect much when it comes to wine. First claster spend less, than other clusters")
    st.markdown("The more the amount spend on wine the less people are on the graph.")


    X2= dataset.iloc[:, [25,24]].values

    kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X2)


    vis_clusters2 = plt.figure(figsize=(10, 6))
    plt.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('Spent_total')
    plt.ylabel('Age')
    plt.legend()
    plt.show()
    st.pyplot(vis_clusters2)

    st.markdown("Data was clustered into 3 parts based on Age and total_spent. Cluster1: spend minimum. Cluste2: spend average. Cluster3: spend the most.")
    st.markdown("Age not much affects on the result.")

    wcss3=[]

    X3= dataset.iloc[:, [2,25]].values

    kmeansmodel = KMeans(n_clusters= 3, init='k-means++', random_state=0)
    y_kmeans= kmeansmodel.fit_predict(X3)


    vis_clusters3 = plt.figure(figsize=(10, 6))
    plt.scatter(X3[y_kmeans == 0, 0], X3[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X3[y_kmeans == 1, 0], X3[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X3[y_kmeans == 2, 0], X3[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.xlabel('Income')
    plt.ylabel('Spent_total')
    plt.legend()
    plt.show()
    st.pyplot(vis_clusters3)

    st.markdown("Here we have interesting relashions. The lower the Income the less the spent_total. The more the indome spent_total also getting bigger.")
    st.markdown("We see direct relations between Income and spent_total.")


    scaler = StandardScaler()
    scaler.fit(dataset)

    dataset = pd.DataFrame(scaler.transform(dataset),columns= dataset.columns)

