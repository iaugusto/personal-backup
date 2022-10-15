import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
import datetime
import pickle
import warnings
import os
from time import time
from tabulate import tabulate
import itertools
import plotly.express as px
import shap
from kmodes.kprototypes import KPrototypes
from tabulate import tabulate

#Sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.ensemble import IsolationForest


class UnsupervisedModelz():
    
    def __init__(self, model_type = 'NONE'):
        
        sns.set_style("darkgrid")
        self.available_models = ['NONE', 'KPROTOTYPES', 'GMM', 'ISOLATION_FOREST']
        self.model_type = model_type.upper()
        self.scaler = None
        self.model = None

        assert self.model_type in self.available_models

        return
    
   

    ###############################################################
    ########### FIT
    ###############################################################

    def fit(self, X, params, verbose = False):
        """
        Main function to call model-specific fit functions

        X: pd.DataFrame
        params: dict of hiperparameters for the model
        verbose: print infos

        return: fitted model (original)
                train info: dictionary containing training info
                e.g.
                {
                'hyperparameters' (dict)  - hiperparameters used
                'n_clusters' - number of clusters
                'metrics' (dict) - metrics used to evaluate the model (cost, davies bouldin, etc.)
                'centroids' - pd.DataFrame of the centroids
                }
        """

        if verbose: 
            print(f'Treinando o modelo {self.model_type.upper()}...')
            t_i = time()


        self.train_info = {}
        self.params = params

        # calls diferent training functions in the format train_{model_type}
        if self.model_type == 'KPROTOTYPES':
            self.model, self.train_info = self.fit_kproto(X, self.params)
        
        elif self.model_type == 'GMM':
            self.model, self.train_info = self.fit_gmm(X, self.params)
        
        elif self.model_type == 'ISOLATION_FOREST':
            self.model, self.train_info = self.fit_if(X, self.params)
        
        
        else: print('não existem modelos para essa configuracao')
        
        # prints
        if verbose: 
            train_time = round((time()-t_i)/60,2)
            print(f'Modelo treinado em {train_time} minutos \n')

        return self.model, self.train_info
    
    
    
    def fit_kproto(self, X, params):
        '''
        Fit Kprototypes model (https://github.com/nicodv/kmodes/blob/master)
        '''
        # make list of categorical columns needed for the model
        categorical_ = list(range(len(self.cat_cols)))
        
        kproto = KPrototypes(**params).fit(X.values, categorical=categorical_)
        pred = kproto.predict(X, categorical=categorical_)
        pred = pd.Series(pred, index = X.index)

        # get centroids
        centroids = kproto.cluster_centroids_
        
        
        centroids = pd.concat([pd.DataFrame(centroids[0]), 
                               pd.DataFrame(centroids[1])],
                               axis=1, 
                               sort=False
        ).T

        
        centroids.columns = X.columns

        # get metrics
        num = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            
        metrics = {}
        metrics['cost'] = kproto.cost_
        # TODO
        # implemet pairwise custom distances
        # linked with distance measure chosen in the model
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        try: 
            metrics['silhouette_score'] = silhouette_score(X.values, pred)
        except: 
            metrics['silhouette_score'] = None
        
        try: 
            metrics['davies_bouldin_score'] = davies_bouldin_score(X.values, pred)
        except: 
            metrics['davies_bouldin_score'] = None

        train_info = {}
        train_info['hyperparameters'] = kproto.get_params()
        train_info['n_clusters'] = kproto.get_params()['n_clusters']
        train_info['metrics'] = metrics
        train_info['centroids'] = centroids
        
        return kproto, train_info
    
    
    def fit_gmm(self, X, params):
        '''
        Fit GMM model 
        (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
        '''

        gmm = GMM(**params).fit(X) 
        
        
        pred = gmm.predict(X)
        pred = pd.Series(pred, index = X.index)

        # get centroids
        centroids = pd.DataFrame(gmm.means_, columns = X.columns)
        
        # get metrics  
        metrics = {}
        metrics['BIC'] = gmm.bic(X)
        metrics['AIC'] = gmm.aic(X)
        try:metrics['silhouette_score'] = silhouette_score(X, pred, metric = 'mahalanobis')
        except: metrics['silhouette_score'] = None
        try: metrics['davies_bouldin_score'] = davies_bouldin_score(X, pred)
        except: metrics['davies_bouldin_score'] = None

        train_info = {}
        train_info['hyperparameters'] = gmm.get_params()
        train_info['n_clusters'] = gmm.get_params()['n_components']
        train_info['metrics'] = metrics
        train_info['centroids'] = centroids
        
        return gmm, train_info


    def fit_if(self, X, params):
        '''
        Fit Isolation Forest model 
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
        '''

        isolation = IsolationForest(**params).fit(X) 

        train_info = {}
        train_info['hyperparameters'] = isolation.get_params()

        
        return isolation, train_info


    def plot_train_info(self, X):
        '''
        Plots train info for each type of model
        * The model needs to be fitted before calling this function

        '''
        assert self.model != None, 'O modelo ainda não foi treinado!'
        assert self.model_type not in ['ISOLATION_FOREST'], f'o modelo {self.model_type} nao suporta essa funcionalidade'

        pred = pd.Series(self.predict(X), index = X.index)

        # Plots to be made
        if self.model_type in ['KPROTOTYPES']:
            self.plot_clusters_population(pred)
            self.plot_tsne(X[self.num_cols], [5, 10, 30, 40], pred.tolist())
            self.plot_categoricals(X[self.cat_cols], pred)
            self.plot_numericals(X, pred)

        if self.model_type in ['GMM']:
            self.plot_clusters_population(pred)
            self.plot_tsne(X[self.num_cols], [5, 10, 30, 40], pred.tolist())
            self.plot_numericals(X, pred)
        

        print('CENTROIDS:\n')
        centroids = self.preprocess_inverse_transform(self.train_info['centroids'])


        print(tabulate(centroids,
                       tablefmt="github",
                       floatfmt = '.4f', 
                       headers="keys", 
                       showindex= True, 
                       stralign = 'center'
                      ))

        sns.heatmap(self.train_info['centroids'], cmap = 'vlag')
        plt.show()

        return centroids


    ###############################################################
    ########### PREDICT
    ###############################################################

    def predict(self, X):
        """
        Main function to call model-specific predict functions

        X: pd.DataFrame

        return: array of predicted values
        """

        # MSF
      

        # calls diferent predict functions in the format predict_{model_type}
        if self.model_type == 'KPROTOTYPES':
            pred = self.model.predict(X, categorical=list(range(len(self.cat_cols))))
        
        elif self.model_type == 'GMM':
            pred = self.model.predict(X)
        
        elif self.model_type == 'ISOLATION_FOREST':
            pred = (
                np.where(self.model.predict(X)<0, 1, 0) , 
                self.model.score_samples(X)
            
            )


        return pred
    



    ###############################################################
    ########### SEARCH
    ###############################################################
    
    
    def parallel_search(self, X, params, min_clusters = 2, max_clusters = 30):
        # TODO
        """
        Main function to call model-specific search functions

        X: pd.DataFrame
        params: dict of hiperparameters for the model
        min_clusters: minimum number of clusters
        max_clusters: maximum number of clusters

        return: search_results (dict)
                e.g.
                {
                number of clusters (str):
                    {
                        'model': model,
                        'train_info': dict of training info
                    }
                }
        """
        print('Nao implementado...')
        exit()

        # MSF
        import concurrent.futures
        assert self.model_type not in ['ISOLATION_FOREST'], 'modelo nao suporta search'


        print(f'Realizando busca com o modelo {self.model_type.upper()}...')
        t_i = time()

        self.search_results = {}
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            secs = list(range(min_clusters,max_clusters+1))
            results = executor.map(do_something, secs)
        
        
        
        
            
            
        for n in range(min_clusters, max_clusters+1):
            time1 = time()

            if self.model_type in ['KPROTOTYPES']:
                params['n_clusters'] = n
            elif self.model_type in ['GMM']:
                params['n_components'] = n


            model, ti = self.fit(X, params, verbose = False)
            
            self.search_results[str(n)] = {'model': model, 'train_info': ti}

            print(f'Fitted n_clusters = {n} ({round(time()-time1, 2)} s)')
        
        

        search_time = round((time()-t_i)/60,2)
        print(f'\nBusca feita em {search_time} minutos')

        return self.search_results



    def search(self, X, params, min_clusters = 2, max_clusters = 30):
        """
        Main function to call model-specific search functions

        X: pd.DataFrame
        params: dict of hiperparameters for the model
        min_clusters: minimum number of clusters
        max_clusters: maximum number of clusters

        return: search_results (dict)
                e.g.
                {
                number of clusters (str):
                    {
                        'model': model,
                        'train_info': dict of training info
                    }
                }
        """

        # MSF
        assert self.model_type not in ['ISOLATION_FOREST'], 'modelo nao suporta search'


        print(f'Realizando busca com o modelo {self.model_type.upper()}...')
        t_i = time()

        self.search_results = {}
        for n in range(min_clusters, max_clusters+1):
            time1 = time()

            if self.model_type in ['KPROTOTYPES']:
                params['n_clusters'] = n
            elif self.model_type in ['GMM']:
                params['n_components'] = n


            model, ti = self.fit(X, params, verbose = False)
            
            self.search_results[str(n)] = {'model': model, 'train_info': ti}

            print(f'Fitted n_clusters = {n} ({round(time()-time1, 2)} s)')
        
        

        search_time = round((time()-t_i)/60,2)
        print(f'\nBusca feita em {search_time} minutos')

        return self.search_results



    def plot_search_results(self):
        
        # create results dataframe
        m = [self.search_results[i]['train_info']['metrics'] for i in self.search_results.keys()]
        
        # generate dataframe
        # columns: metrics, rows: clusters
        results_df  = (
            pd.DataFrame(m)
            .assign(cluster =  self.search_results.keys())
            .set_index('cluster')
        )

        # plot results
        for metric in results_df.columns:
            
            # compute percentage of nans in metric
            nan_perc = results_df[metric].isna().sum()/results_df.shape[0]
            
            if nan_perc > 0.9:
                print(f'contagem de nulos muito alta ({nan_perc*100}%) na metrica {metric}!')
            
            else:
                results_df.plot(y = metric, 
                                kind = 'bar',
                                figsize = (9,4),
                                title = metric
                                )
                plt.show()
            
                
        return results_df



    ###############################################################
    ########### PREPROCESS
    ###############################################################


    def find_numerical_categorical(self, X):
        num = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        num_cols = X.select_dtypes(include=num).columns
        cat_cols = X.columns.drop(num_cols) 

        return list(num_cols), list(cat_cols)


    def preprocess_fit_transform(self, X):
        '''
        Fit preprocessing operation in the dataset
         - MinMaxScaler
         - reorder categorical and numerical columns
                
        '''
        self.num_cols, self.cat_cols = self.find_numerical_categorical(X)
        self.scaler = MinMaxScaler()

        Xnum = X[self.num_cols].copy()
        Xcat = X[self.cat_cols].copy()

        Xscaled = self.scaler.fit_transform(Xnum)
        Xscaled = pd.DataFrame(Xscaled, columns = self.num_cols, index = X.index)

        return pd.concat([Xcat, Xscaled], axis = 1)

    def preprocess_transform(self, X):
        '''
        Aplly transformation with previously fitted objects in dataset.
                
        '''
        Xnum = X[self.num_cols].copy()
        Xcat = X[self.cat_cols].copy()


        Xscaled = self.scaler.transform(Xnum)
        Xscaled = pd.DataFrame(Xscaled, columns = self.num_cols, index = X.index)

        return pd.concat([Xcat, Xscaled], axis = 1)
    
    def preprocess_inverse_transform(self, X):
        '''
        Aplly inverse transformation with previously fitted objects in dataset.
                
        '''
        Xnum = X[self.num_cols].copy()
        Xcat = X[self.cat_cols].copy()


        Xscaled = self.scaler.inverse_transform(Xnum)
        Xscaled = pd.DataFrame(Xscaled, columns = self.num_cols, index = X.index)

        return pd.concat([Xcat, Xscaled], axis = 1)

    ###############################################################
    ########### PLOTS
    ###############################################################

    def plot_tsne(self, X, perplexities = [5, 10, 30, 40], pred = None, **kwargs):


        if self.scaler == None:
            print('escalonando os dados...')
            X = pd.DataFrame(MinMaxScaler().fit_transform(X))

        nrows = int(len(perplexities)/2+0.9999)
        ncols = 2
        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=(15,nrows*7))

        i = 1
        for perplexity in perplexities:

            tsne = TSNE(n_components = 2, random_state = 666, perplexity=perplexity).fit_transform(X)
            df_tsne = pd.DataFrame(tsne)

            plt.subplot(nrows,ncols, i)
            sns.scatterplot(data=df_tsne,
                            x=df_tsne.columns[0], 
                            y=df_tsne.columns[1],
                            alpha = 0.5,
                            hue = pred,
                            palette = 'tab10'
                            
            )
            plt.title(f'TSNE plot - perplexity = {perplexity}', fontsize = 14)
            i+=1
       
        plt.show()

        return


    def plot_pca(self, X):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X = X.select_dtypes(include=numerics)
        print(f'\ncolunas numericas:\n{list(X.columns)}\n')
     
        # plot explanied variance cumsum
        self.scaler = StandardScaler()
        Xscaled = self.scaler.fit_transform(X)
        Xscaled = pd.DataFrame(Xscaled, columns = X.columns, index = X.index)
        
        pca_ = PCA(n_components =  len(X.columns)).fit(Xscaled)

        epr = pca_.explained_variance_ratio_.cumsum()
        x = list(range(len(epr)))
        x_pos = [i for i, _ in enumerate(x)]

        plt.figure(figsize = (15,5))
        plt.bar(x_pos, epr)
        plt.xlabel("Numero de componentes principais", fontsize = 14)
        plt.ylabel("variancia explicada cumulativa", fontsize = 14 )
        plt.title("Resultados PCA", fontsize = 14)
        plt.xticks(fontsize=14)
        plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=14)

        plt.xticks(x_pos, x)

        plt.show()

        # plot 3D graph
        pred = PCA(n_components =  4).fit_transform(Xscaled)
        
        var_ = ['x','y', 'z', 'w']
        a = pd.DataFrame(pred, columns =var_ )

        fig = px.scatter_3d(a, x='x', y='y', z='z',opacity=0.7, color='w')
        fig.show()

        return 

    

    def plot_categoricals(self, X, pred):
        dfcat = X.copy()
        dfcat['cluster_'] = pred.astype(int)
        
        for col in list(dfcat.columns.drop('cluster_')):
            x = pd.get_dummies(dfcat[['cluster_',col]]).groupby('cluster_').sum()
            x = x.div(x.sum(axis=1), axis=0)*100
            x.plot(kind = 'bar',stacked=True, ylim = (0,140), figsize = (12,5), fontsize = 13)
            plt.xlabel('Cluster', fontsize=15)
            plt.ylabel('Fração dos dados (%)', fontsize=15)

            plt.show()

        return

    def plot_numericals(self, X, pred):
        sns.set(rc={"axes.facecolor": (1, 0, 0, 0), "grid.linewidth": 0.5})
        X = self.preprocess_inverse_transform(X)
        for col in list(X[self.num_cols].columns):
            print(f'\n---- {col} -------------------------------------------------------')
            self.plot_fancy_distribuctions(X[col], pred)
            plt.show()
        
        sns.set_style("darkgrid")

        return

    def plot_fancy_distribuctions(self, col, pred):

        k = pd.concat([col, pred], axis=1)
        k.columns = [col.name,'g']

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(k, row="g", hue="g", aspect=7, height=1, palette = pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, col.name, clip_on=False, shade=True, alpha=1, lw=0.2)
        g.map(sns.kdeplot, col.name, clip_on=False, color="w", lw=2)
        g.map(plt.axhline, y=0, lw=1.5, clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)


        g.map(label, col.name)

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        
        return

    def plot_clusters_population(self, pred):
    
        cluster_frac = pred.value_counts()/len(pred)*100
        cluster_frac.plot(kind = 'bar', figsize = (12,5))
        plt.ylabel('Dataset percentage (%)', fontsize = 14)
        plt.xlabel('Cluster number', fontsize = 14)
        plt.tick_params(labelsize=14)
        plt.show()
        


    