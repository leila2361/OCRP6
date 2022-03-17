import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.preprocessing import LabelBinarizer, Normalizer, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import NMF,TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
#from sklearn import model_selection, preprocessing, pipeline
from sklearn import svm, neighbors, manifold
from sklearn.pipeline import make_pipeline
import sklearn.metrics as metrics

from scipy.optimize import linear_sum_assignment

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nbpep8.nbpep8 import pep8


def perplex_coherence(model,corpus, text, dictionary, name_col):
    ''' Return a dataFrame with scores of perplexity and coherence
    Args : 
    model : model to evaluate
    corpus : list
    dictionary : dictionary
    name_col : string
    '''
    # Perplexity score
    perplexity =  model.log_perplexity(corpus)
    # Coherence Score
    coherence_model = CoherenceModel(model=model, 
                                     texts=text, 
                                     dictionary=dictionary, 
                                     coherence='c_v')
    coherence = coherence_model.get_coherence()
    return pd.DataFrame({name_col: [perplexity.round(3),
                                      coherence.round(3)]},
                       index = ['Perplexity', 'Coherence'],
                       dtype = 'float64')
        
    
def topic_cat_matrix(model, corpus, cat_binarized):
    '''For the model, return the matrix topic -categories and a dataframe documents-topics
    Args :
    model : model
    corpus : list
    cat_binarized : categories binarized
    '''
    doc_topic = pd.DataFrame(model\
                             .get_document_topics(corpus,
                                                  minimum_probability=0))
    for topic in doc_topic.columns:
        doc_topic[topic] = doc_topic[topic].apply(lambda x : x[1])
    
    topic_cat = doc_topic.T.dot(cat_binarized)
    return topic_cat,doc_topic

def results(doc_topic, topic_cat, cat_binarized, category) :
    '''return a dataframe with best_topic and predicted categories
    Args :
    doc_topic : data
    topic_cat : matrix
    cat_binarized : binarized categories
    category : categories type string
    '''
    results = pd.DataFrame(category)
    results['best_topic'] = doc_topic.idxmax(axis=1).values
    df_cat_bin = pd.DataFrame(cat_binarized)
    df_dict = dict(
    list(df_cat_bin.groupby(df_cat_bin.index))
    )
    cat_num = []
    for k, v in df_dict.items():
        check = v.columns[(v == 1).any()]
        cat_num.append(check[0])

    results["y_true"] = cat_num
    
    num_cat = []
    for row in results.itertuples():
    
        best_topic = row.best_topic
        row_cat = topic_cat.iloc[best_topic]\
        .sort_values(ascending=False).index[0]
        num_cat.append(row_cat)
    
    results["y_pred"] = num_cat
    return results

def scores_aprf1(results, name_col) :
    '''Return a dataframe of scores : accuracy, precision, recall and f1
    Args:
    results : dataframe
    name_col : string
    '''
    test_accuracy = (metrics.accuracy_score(results['y_true'], results['y_pred']))
    precision = metrics.precision_score(results['y_true'], results['y_pred'], average='weighted')
    recall = metrics.recall_score(results['y_true'], results['y_pred'], average='weighted')
    f1_score = metrics.f1_score(results['y_true'], results['y_pred'], average='weighted')

    return pd.DataFrame({name_col: [test_accuracy,
                              precision,
                              recall,
                              f1_score]},
                              index=['Test Accuracy', 'Precision','Recall','F1_score'],
                              dtype='float64')
    

def plot_top_words(model, feature_names, 
                   n_top_words,  title):
    '''Displaying the plots of the 
    best x words representative of the topics.

    Args:
        model : model
        feature_names : array
        Categories result of the vectorizer (TFIDF ...)
        n_top_words : int
        Number of words for each topic.
        title : string Title of the plot.
   
    '''
    rows = 2
    fig, axes = plt.subplots(rows, 4, 
                             figsize=(30, rows*10), 
                             sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        if(topic_idx < 7):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            bartopic = ax.barh(top_features, weights, height=0.7)
            bartopic[0].set_color('#f48023')
            ax.set_title(f'Topic {topic_idx +1}',
                         fontdict={'fontsize': 20})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=20)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=24)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
def plotCumulativeVariance(co_occurrence_matrix):
    ''' Displaying cumulative variance by components
    Args:
    co_occurence_matrix: matrix
    '''
    max_features = co_occurrence_matrix.shape[1]-1
    svd = TruncatedSVD(n_components=max_features)
    svd_data = svd.fit_transform(co_occurrence_matrix)
    percentage_var_explained = svd.explained_variance_ / np.sum(svd.explained_variance_)
    cum_var_explained = np.cumsum(percentage_var_explained)
    # Plot the TrunvatedSVD spectrum
    plt.figure(1, figsize=(6, 4))
    plt.clf()
    plt.plot(cum_var_explained, linewidth=2)
    plt.axis('tight')
    plt.grid()
    plt.xlabel('n_components')
    plt.ylabel('Cumulative_explained_variance')
    plt.title("Cumulative_explained_variance VS n_components")
    plt.show()

def computeVectors(co_occurrence_matrix, num_components): 
    '''Return projection on nb components of matrix
    by TruncatedSVD
    Args:
    co_occurence : matrix
    num_components : int
    '''
    svd = TruncatedSVD(n_components=num_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    # Returns Transformed matrix of Word-Vectors
    lsa_transform = lsa.fit_transform(co_occurrence_matrix)
    return lsa_transform

def clusters_scores(labels1, labels2, name_col):
    '''Return a dataframe of clustering scores :
    Homogeneity, Completeness, V-measure, ARI
    Args:
    'labels1 : list
    labels2 : list
    name_col : string
    '''
    homo = metrics.homogeneity_score(labels1, labels2)
    comp = metrics.completeness_score(labels1, labels2)
    v_mea = metrics.v_measure_score(labels1, labels2)
    ari = metrics.adjusted_rand_score(labels1, labels2)
    
    return  pd.DataFrame({name_col: [homo,
                              comp,
                              v_mea,
                              ari]},
                              index=['Homogeneity',
                                     'Completeness',
                                     'V_measure','ARI'],
                              dtype='float64')

def plot_TSNE(X, hue1, hue2, colors, titre):
    '''Figure of TSNE projection 2D
    Args :
    X : ndarray or data to project
    hue1 : list or data column
    hue2 : list or data column
    colors : list for the colors
    titre : string
    '''
    Y = manifold.TSNE().fit_transform(X)

    fig = plt.figure(figsize=(10, 4))
    plt.suptitle('Predictions vs Reelles pour '+ titre, fontsize=14)
    palet = sns.color_palette('hls',7)
    
    colori = {}
    for k in range(7) :
        colori[colors[k]] = palet[k]
    
    ax = fig.add_subplot(1,2,1)
    ax = sns.scatterplot(x=Y[:, 0], y=Y[:, 1],
                     hue=hue1,
                     palette=colori, 
                     legend=False)
    plt.title ('categories predites')

    ax = fig.add_subplot(1,2,2)
    ax = sns.scatterplot(x=Y[:, 0], y=Y[:, 1],
                     hue=hue2,
                     palette=colori)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.85, box.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.6,0.5), ncol=1)
    plt.title('categories reelles')

    plt.show()


def model_benchmark(x_train, y_train, x_test, y_test, models):
    """Function to compare a list of model with a dataset

    Args :
    x : arrays or pd.dataFrame
    y : one dimension array or pd.Series
    models : a list of model included in the model_benchmark could be chose

    Return : 
    a dataframe with best param, mse and r square for each model tested

    """
    # Create a dF to stock results of each model by column
    model_performance = pd.DataFrame(dtype='float64')

    # List to stock param of each model
    param = []
    yp = []

    for i in models:
        model = i
      
        # Grid search with grid param in each model function
        grids = GridSearchCV(i['model'], param_grid=i['param'], cv=3, n_jobs=-1)

        # Fit the model / then predict y value
        grids.fit(x_train, y_train)
        y_pred = grids.predict(x_test)

        # Best params in training
        param.append([i['model_name'], grids.best_params_])
        yp.append([i['model_name'], y_pred])
        
        # Scoring
        test_accuracy = (metrics.accuracy_score(y_test, y_pred))
        macro_precision = metrics.precision_score(y_test, y_pred, average='macro')
        macro_recall = metrics.recall_score(y_test, y_pred, average='macro')
        macro_f1_score = metrics.f1_score(y_test, y_pred, average='macro')

        # Create a dF to display all the score results obtained by ML model
        model_serie = pd.DataFrame({i['model_name']:\
                                   [test_accuracy.round(3),
                                    macro_precision,
                                    macro_recall,
                                   macro_f1_score]},
                                   index=['Test Accuracy', 'Precision','Recall','F1_score'],
                                   dtype='float64')
        
        model_performance = pd.concat([model_serie, model_performance], axis=1)     

    # Output dict
    return {'perf': model_performance,
        'param': param,
        'y_pred': yp}

def rfc(n_estim):
    grid_param1 = {'n_estimators': n_estim}
    
    return {'model_name': 'rfc',
            'model': RandomForestClassifier(),
            'param': grid_param1}

def svc(penalty, loss):
    grid_param3 = {'penalty': penalty,
                   'loss': loss,
                   'class_weight' : ['balanced']}

    return {'model_name': 'svc',
            'model': svm.LinearSVC(),
            'param': grid_param3}

def lr(c):
    grid_param4 = {'multi_class': ['multinomial'],
                   'solver': ['lbfgs'],
                   'C': c,
                   'class_weight': ['balanced']}
    
    return {'model_name': 'lr',
            'model': LogisticRegression(),
            'param': grid_param4}

def knn(n_neigh, weight):
    grid_param5 = {'n_neighbors': n_neigh,
                   'weights': weight}

    return {'model_name': 'knn',
            'model': neighbors.KNeighborsClassifier(),
            'param': grid_param5}

def multinb(alpha) :
    grid_param6 = {'alpha': alpha}
    return {'model_name': 'multinb',
            'model': MultinomialNB(),
           'param': grid_param6}

def show_images_cat(data, category, num_sample):
    """ Displaying the first 
    n images of a data passed as an argument.      
    Args :
    data : dataset
    category : string 
    num_sample : integer
        Number of picture to show
    """
    fig = plt.figure(figsize=(15,8))
    fig.patch.set_facecolor('#343434')
    plt.suptitle("{}".format(category), y=.9,
                 color="white", fontsize=22)
    temp = data['img']
    for i in range(num_sample):
        j = np.random.randint(1, len(temp)-4)
        plt.subplot(2, 5, i+1)
        plt.imshow(temp[i+j])
        plt.axis('off')
    plt.show()  

def plot_histogram(init_img, convert_img):
    """Display the initial
    and converted images according to a certain
    colorimetric format as well as the histogram
    of the latter. 

    Args :
    init_img : list
        init_img[0] = Title of the init image
        init_img[1] = Init openCV image
    convert_img : list
        convert_img[0] = Title of the converted
        convert_img[1] = converted openCV image
    """
    hist, bins = np.histogram(
                    convert_img[1].flatten(),
                    256, [0,256])
    # Cumulative Distribution Function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Plot histogram
    fig = plt.figure(figsize=(25,6))
    plt.subplot(1, 3, 1)
    plt.imshow(init_img[1])
    plt.title("{} Image".format(init_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 2)
    plt.imshow(convert_img[1])
    plt.title("{} Image".format(convert_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 3)
    plt.plot(cdf_normalized, 
             color='r', alpha=.7,
             linestyle='--')
    plt.hist(convert_img[1].flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title("Histogram of convert image", color="#343434")
    plt.suptitle("Histogram and cumulative "\
                 "distribution for test image",
              color="black", fontsize=22, y=.98)
    plt.show()

def conf_matr_max_diago(true_cat, clust_lab, normalize=False):
    '''Return confusion matrix
    Args:
    true_cat : data column or list
    clust_label : data column or list of predictions
    '''

    ### Count the number of articles in eact categ/clust pair
    cross_tab = pd.crosstab(true_cat, clust_lab,
                            normalize=normalize)

    ### Rearrange the lines and columns to maximize the diagonal values sum
    # Take the invert values in the matrix
    func = lambda x: 1/(x+0.0000001)
    inv_func = lambda x: (1/x) - 0.0000001
    funct_trans = FunctionTransformer(func, inv_func)
    inv_df = funct_trans.fit_transform(cross_tab)

    # Use hungarian algo to find ind and row order that minimizes inverse
    # of the diag vals -> max diag vals
    row_ind, col_ind = linear_sum_assignment(inv_df.values)
    inv_df = inv_df.loc[inv_df.index[row_ind],
                        inv_df.columns[col_ind]]

    # Take once again inverse to go back to original values
    cross_tab = funct_trans.inverse_transform(inv_df)
    result = cross_tab.copy(deep='True')

    if normalize == False:
        result = result.round(0).astype(int)

    return result

def categ_identificator(df_labels, true_cat=None):
    '''Transform numeric labels to string
    Args:
    df_labels : data with columns are predicted labels
    '''

    # takes the first columns as true category if true_cat not specified
    if true_cat is None:
        true_cat_ser = df_labels.iloc[:,0]
        df_labels = df_labels.iloc[:,1:]
    else:
        true_cat_ser = true_cat

    df_ = pd.DataFrame()

    # loop on all the lists of labels
    for col in df_labels.columns:
        # generate a confusion matrix that maximizes the diagonal
        cm = conf_matr_max_diago(true_cat_ser, df_labels[col],
                                normalize=False)
        # generate a translation dictionary to apply to the column col
        transl = dict(zip(cm.columns, cm.index))
        df_[col] = df_labels[col].map(transl)

    # if true_cat was the first columns, reinsert
    if true_cat is None:
        df_.insert(0, true_cat_ser.name, true_cat_ser)
    return df_