
# machine learning constants -----------------------------------------------
MODEL_DICT = {
    "Naive Bayes Classifier": "naive_bayes",
    "K Nearest Neighbour": "knn",
    "Logistic Regression": "log_reg",
    "Linear Regression": "lin_reg",
    "Decision Tree": "decision_tree",
    "clustering-kmeans": "kmeans",
    "Hidden Markov Model": "hmm",
    "Principal Component Analysis": "pca",
    "Perceptron": "perceptron",
    "Backpropogation": "backprop",
    "Neural Networks": "neural_nets",
    "Computer Vision": "comp_vision",
    "Natural Language Processing": "nlp",
    "Autoencoder": "ae",
}

# classification machine learning constants -----------------------------------------------
CLASSIFICATION_DICT = {
    "workclass_dict": [
        "Private", "Local-gov", "?", "Self-emp-not-inc", "Federal-gov",
        "State-gov", "Self-emp-inc", "Without-pay", "Never-worked"
    ],
    "education_dict": [
        "11th", "HS-grad", "Assoc-acdm", "Some-college", "10th", "Prof-school",
        "7th-8th", "Bachelors", "Masters", "Doctorate", "5th-6th", "Assoc-voc",
        "9th", "12th", "1st-4th", "Preschool"
    ],
    "marital_status_dict": [
        "Never-married", "Married-civ-spouse", "Widowed", "Divorced",
        "Separated", "Married-spouse-absent", "Married-AF-spouse"
    ],
    "occupation_dict": [
        "Machine-op-inspct", "Farming-fishing", "Protective-serv", "?",
        "Other-service", "Prof-specialty", "Craft-repair", "Adm-clerical",
        "Exec-managerial", "Tech-support", "Sales", "Priv-house-serv",
        "Transport-moving", "Handlers-cleaners", "Armed-Forces"
    ],
    "relationship_dict": [
        "Own-child", "Husband", "Not-in-family", "Unmarried", "Wife",
        "Other-relative"
    ],
    "race_dict": [
        "Black", "White", "Asian-Pac-Islander",
        "Other", "Amer-Indian-Eskimo"
    ],
    "gender_dict": ["Male", "Female"],
    "native_country_dict": [
        "United-States", "?", "Peru", "Guatemala", "Mexico",
        "Dominican-Republic", "Ireland", "Germany", "Philippines", "Thailand",
        "Haiti", "El-Salvador", "Puerto-Rico", "Vietnam", "South", "Columbia",
        "Japan", "India", "Cambodia", "Poland", "Laos", "England", "Cuba",
        "Taiwan", "Italy", "Canada", "Portugal", "China", "Nicaragua",
        "Honduras", "Iran", "Scotland", "Jamaica", "Ecuador", "Yugoslavia",
        "Hungary", "Hong", "Greece", "Trinadad&Tobago",
        "Outlying-US(Guam-USVI-etc)", "France", "Holand-Netherlands"
    ],
    "income_dict": ["<=50K", ">50K"]
}

# regression machine learning constants -----------------------------------------------
REGRESSION_DICT = {}

CLUSTER_DICT = {
    "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
}
