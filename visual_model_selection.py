import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,precision_score,accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
import scipy
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import cm
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
from bioinfokit.visuz import cluster
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


def MC_visual_density_plot(X_DR, Y_list, export_file=True):
    """
    Figure 5 in the paper performing pairwise comparison from a list of classifiers via density plot.
    
    Parameters
    ----------
    X_DR : a reduced dimensionality data matrix X in 2D represented by a pandas dataframe of shape (n samples, 2)
    Y_list : a list of tuples that consist of model name and model predictions
    """
    matrix=X_DR.copy()
    col=matrix.columns.tolist()
    
    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    
    for i1 in range(len(Y_list)):
        for i2 in range(i1+1,len(Y_list)):
            model1_name = Y_list[i1][0]
            model2_name = Y_list[i2][0]
            classes1 = np.unique(Y_list[i1][1])
            classes2 = np.unique(Y_list[i2][1])

            df2 = matrix[model1_name].astype(str) + matrix[model2_name].astype(str)

            fig = plt.figure(figsize=[10,8])
            sns.kdeplot(data=matrix, x=matrix.DR1, y=matrix.DR2, hue=df2, fill=True, alpha=.6)
            plt.title(f'MC visual density plot {model1_name} vs. {model2_name}')
            
            if export_file is False:
                pass
            else:
                plt.savefig(f'MC_visual_density_plot_{model1_name}_{model2_name}.png', dpi=300)


                

def MC_visual_confusion_matrix(X_DR, Y_list, export_file=True):
    """
    Figure 6 in the paper performing pairwise comparison from a list of classifiers via visual confusion matrix.

    Parameters
    ----------
    X_DR : a reduced dimensionality data matrix X in 2D represented by a pandas dataframe of shape (n samples, 2)
    Y_list : a list of tuples that consist of model name and model predictions
    """

    matrix=X_DR.copy()
    col=matrix.columns.tolist()
    
    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    
    for i1 in range(len(Y_list)):
        for i2 in range(i1+1,len(Y_list)):
            model1_name = Y_list[i1][0]
            model2_name = Y_list[i2][0]
            classes1 = np.unique(Y_list[i1][1])
            classes2 = np.unique(Y_list[i2][1])

            #figure4a
            fig, axs = plt.subplots(len(classes1),len(classes2), figsize = (10,8),sharex=True, sharey=True)
            color=['b','r','goldenrod','g']

            c=-1
            for i in range(len(classes1)):
                for j in range(len(classes2)):
                    c+=1
                    relevant_observations = (matrix[model1_name]==classes1[i]) & (matrix[model2_name]==classes2[j])
                    axs[i,j].plot(X_DR.iloc[:,0][relevant_observations], X_DR.iloc[:,1][relevant_observations], marker='o', linestyle='', color =color[c], alpha=0.3)    

                    axs[0,0].set_title('0')
                    axs[0,1].set_title('1')

                    axs[0,0].set_ylabel('0')
                    axs[1,0].set_ylabel('1')

            fig.text(0.5,0.95, model2_name, ha="center", va="center")
            fig.text(0.05,0.5, model1_name , ha="center", va="center", rotation=90)
            plt.savefig(f'MC_visual_confusion_matrix_split_{model1_name}_{model2_name}.png', dpi=300)

            #figure4b
            fig = plt.figure(figsize=[8,6]) 
            c=0
            for i in range(len(classes1)):
                for j in range(len(classes2)):
                    if classes1[i]!=classes2[j]:
                        c+=1
                        relevant_observations = (matrix[model1_name]==classes1[i]) & (matrix[model2_name]==classes2[j])
                        plt.plot(X_DR.iloc[:,0][relevant_observations], X_DR.iloc[:,1][relevant_observations], marker='o', linestyle='', color =color[c], label=[(model1_name, classes1[i]),(model2_name, classes2[j])], alpha=0.3)
                        plt.legend()
            if export_file is False:
                pass
            else:
                plt.savefig(f'MC_visual_confusion_matrix_combined_{model1_name}_{model2_name}.png', dpi=300)






def MC_biplot(X_pca, pca_model, Y_list, features, export_file=True):

    """
    Figure 7 in the paper performing pairwise comparison from a list of classifiers via Biplot.
    
    Parameters
    ----------
    X_pca : a PCA reduced dimensionality data matrix X in 2D represented by a pandas dataframe of shape (n samples, 2)
    pca_model: model of pca
    Y_list : a list of tuples that consist of model name and model predictions
    ----------
    cluster.biplot is from the following website: https://reneshbedre.github.io/blog/howtoinstall.html
    """ 
    
    matrix=X_DR.copy()
    col=matrix.columns.tolist()
    
    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    

    loadings = pca_model.components_
    pca_model.explained_variance_

    for i1 in range(len(Y_list)):
        for i2 in range(i1+1,len(Y_list)):
            model1_name = Y_list[i1][0]
            model2_name = Y_list[i2][0]
            classes1 = np.unique(Y_list[i1][1])
            classes2 = np.unique(Y_list[i2][1])

            df2 = matrix[model1_name].astype(str) + matrix[model2_name].astype(str)
            relevant_observations = (matrix[model1_name]!=matrix[model2_name])
            target = df2[relevant_observations]
            df3 = matrix[['DR1','DR2']][relevant_observations]
            pca_scores = df3.to_numpy()
            
            print('MC_biplot '+ model1_name + ' vs. ' + model2_name)
            if export_file is False:
                cluster.biplot(cscore=pca_scores, loadings=loadings, labels=features,
                               var1=round(pca_model.explained_variance_ratio_[0]*100, 2),
                               var2=round(pca_model.explained_variance_ratio_[1]*100, 2), colorlist=target,
                               show=True, dim=(10,8), valphadot=0.3, dotsize=15, arrowlinewidth=1.5,
                               arrowcolor='orange', axlabelfontsize=15, figtype='png',
                               colordot=['#e6b619', '#d65418']) #'#42b86d', '#1884d6'
            else:
                cluster.biplot(cscore=pca_scores, loadings=loadings, labels=features,
                               var1=round(pca_model.explained_variance_ratio_[0]*100, 2),
                               var2=round(pca_model.explained_variance_ratio_[1]*100, 2), colorlist=target,
                               show=True, dim=(10,8), valphadot=0.3, dotsize=15, arrowlinewidth=1.5,
                               arrowcolor='orange', axlabelfontsize=15, figtype='png',
                               colordot=['#e6b619', '#d65418']) #'#42b86d', '#1884d6'
            
                cluster.biplot(cscore=pca_scores, loadings=loadings, labels=features,
                               var1=round(pca_model.explained_variance_ratio_[0]*100, 2),
                               var2=round(pca_model.explained_variance_ratio_[1]*100, 2), colorlist=target,
                               show=False, dim=(10,8), valphadot=0.3, dotsize=15, arrowlinewidth=1.5,
                               arrowcolor='orange', axlabelfontsize=15, figtype='png',
                               colordot=['#e6b619', '#d65418']) #'#42b86d', '#1884d6'
                



def MC_heatmap_of_prediction(Y_list, export_file=True):
    """
    Figure 3 in the paper performing pairwise comparison from a heatmap of model prediction agreement level.
    
    Parameters
    ----------
    Y_list : a list of tuples (first tuple as true labels, model predictions follow) that consist of model name and model predictions
    """ 
    
    matrix=pd.DataFrame()
    col=[]
    
    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    
    n = matrix.shape[0]
    m = matrix.shape[1]
    res = np.zeros((m,m))
    for i1 in range(matrix.shape[1]):
        for i2 in range(matrix.shape[1]):
            acc = sum(matrix.iloc[:,i1]==matrix.iloc[:,i2])/n
            res[i1,i2]=acc

    res = pd.DataFrame(data=res, columns=matrix.columns.values, index=matrix.columns.values)
    res=res.round(2)

    fig = plt.figure(figsize=[12,10])
    ax = sns.heatmap(res, annot=True)
    
    if export_file is False:
        pass
    else:
        plt.savefig(f'MC_heatmap_of_prediction.png', dpi=300)
        



def MC_cluster_analysis(X, Y_list, n_cl=10 , export_file=True):
    """
    Figure 9 VCX in the paper for cluster analysis

    Parameters
    ----------
    X : a data matrix represented by a pandas dataframe of shape (n samples, p features)
    Y_list : a list of tuples (first tuple as true labels, model predictions follow) that consist of model_name and model_prediction
    """
    kmeans = KMeans(n_clusters=n_cl, random_state=10).fit(X)

    matrix=pd.DataFrame()
    col=[]

    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    matrix['cluster']=kmeans.labels_

    results=[]
    for i in range(n_cl):
        dfc=matrix.loc[(matrix['cluster'] ==i)]
        mod = dfc[models]
        y_true= dfc.iloc[:,0].tolist()
        for column in mod.columns:
            y_pred=dfc[column].tolist()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            res = {}
            res['cluster']=i
            res['Model']=column
            res['true positive rate'] = tp/(tp+fn)
            res['false positive rate'] = fp/(fp+tn)
            res['precision']=precision
            res['recall'] = recall
            results.append(res)

    matrice = pd.DataFrame(results)

    metrics =['true positive rate', 'false positive rate', 'precision', 'recall']
    for metric in metrics:
        model= pd.DataFrame(models, columns =['Model'])
        col=['Model']
        for i, name in matrice.groupby('cluster'):
            name = name.reset_index(drop=True)
            tpmatrix=name[[metric]]
            model=pd.concat([model, tpmatrix], axis=1, ignore_index=True)
            col.append('cluster' + str(i))

        model.columns=col
        model=model.set_index('Model')
        model=model.round(2)

        fig = plt.figure(figsize=[12,10])
        ax = sns.heatmap(model, annot=True)

        ax.set_title(metric, fontsize=10, fontweight="bold") 
        ax.xaxis.tick_top()
        if export_file is False:
            pass
        else:
            plt.savefig(f'VCX_cluster_analysis_{metric}.png', dpi=300)




def MC_simple_dr_comparison(X_DR, Y_list, export_file=True):
    
    """
    Figure 10 in the paper performing pairwise comparison from a list of classifiers by ploting models next to each other.
    The first subfigure shows the true y labels and serves as the baseline.
    Yellow markers represent observations with label (or prediction) 1
    Blue markers represent observations with label (or prediction) 0.
    
    Parameters
    ----------
    X_DR : a reduced dimensionality data matrix X in 2D represented by a pandas dataframe of shape (n samples, 2)
    Y_list : a list of tuples (first tuple as true labels, model predictions follow) that consist of model name and model predictions
    """ 

    matrix=X_DR.copy()
    col=matrix.columns.tolist()
    classifications=[]

    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])

        class_label=np.unique(Y_list[i][1]).tolist()
        classifications.append(class_label)

    matrix.columns=col

    fig = plt.figure(figsize=[12,12])
    for j2 in range(2,matrix.shape[1]):
        model=matrix.columns[j2]
        cl=classifications[j2-2]
        y=matrix[model]
        columnname=matrix.columns[j2]
        ax = fig.add_subplot(4,int(len(classifications)/4),j2-1)

        target_names = [model+' pos', model+' neg']
        colors = ['gold','steelblue']
        lw = 2

        for color, i, target_name in zip(colors, cl, target_names):
            ax.scatter(X_DR.iloc[:,0][y == i], X_DR.iloc[:,1][y == i], color=color, alpha=.3, lw=lw,label=target_name)
            ax.legend()
        fig.tight_layout(h_pad=2, w_pad=1)
    plt.show()

    if export_file is False:
        pass
    else:
        plt.savefig(f'MC_simple_dr_comparison.png', dpi=300)


        


def MC_scatterplot_prediction(Y_list, color_indices=None, colors=None, export_file=True):
    """
    Figure 11 in the paper performing pairwise comparison via plotting PCA dimensionality reduction on model predictions
    Models locate close to each other have similar predictions
    Next to model names are prediction accuracies
    
    Parameters
    ----------
    Y_list : a list of tuples that consist of model_name, model_prediction, and test_accuracy
    color_indices: a list of indices for the colors
    colors: a list of colors
    """
    matrix=pd.DataFrame()
    col=[]

    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col

    df1=matrix.T
    indexnames=df1.index
    X = df1.values[:, :]
    pca = PCA(n_components=2,random_state=20)
    DR = pca.fit_transform(X)
    DR = pd.DataFrame(DR,columns=['DR1','DR2'], index=indexnames)

    fig = plt.figure(figsize=[10,8])

    if color_indices is None or colors is None:
        plt.scatter(x=DR['DR1'], y=DR['DR2'])
        for i, txt in enumerate(DR.index):
            plt.annotate(txt, (DR['DR1'][i]-0.3, DR['DR2'][i]), ha='right')
            plt.annotate(Y_list[i][2], (DR['DR1'][i]+0.3, DR['DR2'][i]), ha='left')
        plt.show()
    else:
        plt.scatter(x=DR['DR1'], y=DR['DR2'], c=color_indices, cmap=matplotlib.colors.ListedColormap(colors))
        for i, txt in enumerate(DR.index):
            plt.annotate(txt, (DR['DR1'][i]-0.3, DR['DR2'][i]), ha='right')
            plt.annotate(Y_list[i][2], (DR['DR1'][i]+0.3, DR['DR2'][i]), ha='left')
        plt.show()
    
    if export_file is False:
        pass
    else:
        plt.savefig(f'MC_scatterplot_prediction.png', dpi=300)




def MC_scatterplot_confusion(Y_list, color_indices=None, colors=None, export_file=True):
    """
    Figure 13 in the paper performing pairwise comparison via plotting PCA dimensionality reduction on models' confusion matrices
    Models locate close to each other have similar confusion matrices
    Next to confusion matrices names are model prediction accuracies
    
    Parameters
    ----------
    Y_list : a list of tuples (first tuple as true labels, model predictions follow) that consist of model_name, model_prediction, and test_accuracy
    color_indices: a list of indices for the colors
    colors: a list of colors
    """
    matrix=pd.DataFrame()
    col=[]

    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col
    matrix.columns = [str(c) + '_vs_Y' for c in matrix.columns]

    for i in range(1,matrix.shape[1]):
        matrix.iloc[:,i] = matrix.iloc[:,i].astype(str) + matrix.iloc[:,0].astype(str)
    matrix['Y_vs_Y']= matrix['Y_vs_Y'].astype(str).str.repeat(2)  

    confusion=[]
    for j in range(0,matrix.shape[1]):
        df1 = pd.DataFrame(matrix.iloc[:,j].value_counts())
        confusion.append(df1)
        conf=pd.concat(confusion, axis=1).T
        conf[np.isnan(conf)]  = 0
        conf=conf.round().astype(int)

    indexnames=conf.index
    X = conf.values[:, :]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2,random_state=20)
    
    pca.fit(X)
    DR = pca.transform(X)
    DR = pd.DataFrame(DR,columns=['DR1','DR2'], index=indexnames)
    
    fig = plt.figure(figsize=[10,8])
        
    if color_indices is None or colors is None:
        plt.scatter(x=DR['DR1'], y=DR['DR2'])
        for i, txt in enumerate(DR.index):
            plt.annotate(txt, (DR['DR1'][i], DR['DR2'][i]), ha='right')
            plt.annotate(Y_list[i][2], (DR['DR1'][i]+0.1, DR['DR2'][i]), ha='left')            
            fig.tight_layout(h_pad=2, w_pad=1)
        plt.show()
    else:
        plt.scatter(x=DR['DR1'], y=DR['DR2'], c=color_indices, cmap=matplotlib.colors.ListedColormap(colors))
        for i, txt in enumerate(DR.index):
            plt.annotate(txt, (DR['DR1'][i], DR['DR2'][i]), ha='right')
            plt.annotate(Y_list[i][2], (DR['DR1'][i]+0.1, DR['DR2'][i]), ha='left')
            fig.tight_layout(h_pad=2, w_pad=1)
        plt.show()
    
    if export_file is False:
        pass
    else:
        plt.savefig(f'MC_scatterplot_confusion.png', dpi=300)




def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


def MC_hierarchical_tree(Y_list, export_file=True):
    """
    Figure 14 in the paper performing model comparison via hierarchical clustering by prediction.
    Models in the same cluster have similar predictions
    
    Parameters
    ----------
    Y_list : a list of tuples that consist of model_name and model_prediction
    ----------
    plot_dendrogram is from the following website: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """ 
        
    matrix=pd.DataFrame()
    col=[]

    for i in range(len(Y_list)):
        matrix=pd.concat([matrix, pd.DataFrame(Y_list[i][1])], axis=1, ignore_index=True)
        col.append(Y_list[i][0])
    matrix.columns=col

    clf = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clf = clf.fit(matrix.T)
    fig = plt.figure(figsize=[12,5])
    # plot the top three levels of the dendrogram
    plot_dendrogram(clf, truncate_mode=None, labels=matrix.columns)
    plt.show()
    
    if export_file is False:
        pass
    else:
        plt.savefig(f'MC_hierarchical_tree.png', dpi=300)
