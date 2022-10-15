import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

from dython._private import convert, remove_incomplete_samples, replace_nan_with_value
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import KFold
from minepy import MINE
from collections import Counter

class FeatureSelection:
   
    ####################################################################################################
    ########################################## FILTERS METHODS #########################################
    ####################################################################################################
    
    def remove_constant_features(self, X):
        """Detect and remove constant features

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with independent features to be analyze

        Returns
        -------
        df : pandas.core.frame.DataFrame
            New dataframe with constants features removed
        drop_features : list
            List of removed features
        """

        a = list(X.columns)
        X = X.loc[:, (X != X.iloc[0]).any()]
        b = list(X.columns)
        constant_features = list(set(a).difference(b))
        
        if len(constant_features) > 0:
            print('Constant features removed are: {}'.format(constant_features))
        else:
            print('There are no constant features')

        return X, constant_features
    
    def remove_quasi_constant_features(self, X, threshold = 0.01):
        """Detect and remove quasi-constant features based on threshold passed

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with independent features to be analyze
        threshold : int
            Threshold used on variance to determine quasi-constant features (default is
        False)

        Returns
        -------
        X : pandas.core.frame.DataFrame
            New dataframe with quasi-constants features removed
        quasi_constant_features : list
            List of removed features
        """

        quasi_constant_features = [
            feat for feat in X.columns if X[feat].var() < threshold
        ]

        #Remove Quasi constant features duplicated from dataframe
        X.drop(labels=quasi_constant_features, axis=1, inplace=True)
        
        if len(quasi_constant_features) > 0:
            print('Quasi constant features removed are: {}'.format(quasi_constant_features))
        else:
            print('There are no quasi-constant features')

        return X, quasi_constant_features

    def remove_duplicated_features(self, X):
        """Detect and remove duplicated features

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with independent features to be analyze

        Returns
        -------
        X : pandas.core.frame.DataFrame
            New dataframe with quasi-constants features removed
        pairs_duplicated_feat : list
            List of removed features
        """
        
        duplicated_feat = []
        pairs_duplicated_feat = []
        print("Starting process to remove duplicate features")
        for i in range(0, len(X.columns)):
            '''
            if i % 10 == 0:  # this helps me understand how the loop is going
                print(i)
            '''
            col_1 = X.columns[i]

            for col_2 in X.columns[i + 1:]:
                if X[col_1].equals(X[col_2]):
                    pairs_duplicated_feat.append(tuple([col_1, col_2]))
                    duplicated_feat.append(col_2)
        print("Completed process for removing duplicate assets")

        #Remove features duplicated from dataframe
        X.drop(labels=duplicated_feat, axis=1, inplace=True)
        
        if len(duplicated_feat) > 0:
            print('Duplicate features removed are: {}'.format(duplicated_feat))
        else:
            print('There are no Duplicated features')

        return X, pairs_duplicated_feat
    
    def calculate_conditional_entropy(self, x, y):
        """Calculates the conditional entropy of x given y: S(x|y)

        Parameters
        ----------
        x : list / NumPy ndarray / Pandas Series
            A sequence of measurements
        y : list / NumPy ndarray / Pandas Series
            A sequence of measurements

        Returns
        -------
        conditional_entropy : float
            Conditional entropy computed from parameter x and y
        """

        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x,y)))
        total_occurrences = sum(y_counter.values())
        conditional_entropy = 0.0

        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences

            conditional_entropy += p_xy * np.log(p_y/p_xy)

        return conditional_entropy

    def calculate_entropy(self, x):
        """Calculates entropy

        Parameters
        ----------
        x : pandas.core.frame.Series
            A sequence of categorical measurements

        Returns
        -------
        entropy : float
            Entropy computed from parameter x
        """
        n_labels = len(x)

        if n_labels <= 1:
            return 0

        value, counts = np.unique(x, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        entropy = 0.

        # Compute entropy
        base = np.e
        for i in probs:
            entropy -= i * np.log(i)

        return entropy

    def calculate_theilsu(self, x, y):
        """Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association

        Parameters
        ----------
        x : pandas.core.frame.Series
            A sequence of categorical measurements
        y : pandas.core.frame.Series
            A sequence of categorical measurements

        Returns
        -------
        uncertainty_coefficient : float
            Uncertainty coefficient computed from parameter x and y, also know as Theil's U
        """

        x_entropy = self.calculate_entropy(x)

        if x_entropy == 0:
            return 1
        else:
            conditional_entropy = self.calculate_conditional_entropy(x, y)
            uncertainty_coefficient = (x_entropy - conditional_entropy) / x_entropy
            return uncertainty_coefficient

    def remove_high_theilsu(self, X, category_features=[], threshold = .95):
        """Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical
        association and remove high related features that tends to bring the same information.
        This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y
        provides no information about x, and 1 means y provides full information about x.
        This is an asymmetric coefficient: U(x,y) != U(y,x)

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with categorical features to be analyze
        threshold : int
            Threshold that uses VIF scores to determine multicollinearity on independent features
            (default is 10)

        Returns
        -------
        good_features: list
            List of independent features that brings new information to prediction and attend the
            threshold
        bad_features: list
            List of independent features that brings redundant information to prediction following
            the threshold
        pair_features: list
            List of tuple that brings the feature that should be removed and the feature that brings
            the same information, or the feature that explain the first to be removed
        theilsu_matrix
            Computed matrix of uncertainty coefficient for each features
        """
        if len(category_features) <= 0:
            print('A list of the columns to be computed must be sent')
            return False

        columns = category_features
        theilsu_matrix = pd.DataFrame(index=columns, columns=columns)

        # Compute Theils'U Matrix
        for i in range(0,len(columns)):
            for j in range(0,len(columns)):
                theilsu_matrix.loc[columns[i], columns[j]] = self.calculate_theilsu(X[columns[i]], X[columns[j]])

        # Plot result matrix
        theilsu_matrix.fillna(value=np.nan,inplace=True)
        ax = plt.axes()
        ax.set_title("Theil's U Heatmap", fontsize=16)
        sns.heatmap(theilsu_matrix, annot=True, fmt='.2f')
        plt.show()

        bad_features = []
        pair_features = []
        # Identify features that are higher than treshold
        for i in range(0, len(category_features)):
            for j in range(0, len(category_features)):
                if theilsu_matrix.iloc[i, j] >= threshold and category_features[i] != category_features[j]:
                    bad_features.append(category_features[i])
                    pair_features.append(tuple([category_features[i], category_features[j]]))

        good_features = list(set(category_features) - set(bad_features))
        return good_features, bad_features, pair_features, theilsu_matrix



    def theilsu_y(self, X, y):
        """
        Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical
        between independent and dependent variables 
        
        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with categorical features to be analyze
        y : pandas.core.frame.Series
            Series with dependent variable measuments
        
        Returns
        -------
        theilsu_matrix
            Computed matrix of uncertainty coefficient for each features
        """

        assert isinstance(y, pd.Series), 'não é uma Series!'

        columns = X.columns

        theilsu_matrix = pd.DataFrame(index=['target'], columns=X.columns)

        for j in range(0,len(columns)):
            theilsu_matrix.loc[:,columns[j]] = self.calculate_theilsu(y, X[columns[j]])

        theilsu_matrix.fillna(value=np.nan, inplace=True)
        plt.figure(figsize=(len(columns), 1))
        sns.heatmap(theilsu_matrix, annot=True, fmt='.2f')
        plt.show()

        theilsu_matrix = theilsu_matrix.T
        theilsu_matrix.sort_values(by=['target'], ascending=False, inplace=True)

        return theilsu_matrix

    ####################################################################################################
    ########################################## WRAPPER METHODS #########################################
    ####################################################################################################
    
    def print_info(self, vif_table, num_original_columns, features_selected, features_removed):
        print("Removing multicollinearity through VIF(Variance Inflation Factor)")
        print("RESULTS:")
        print("=================================================================")
        print("Original number of columns: " + str(num_original_columns))
        print("Number of columns after feature selection: " + str(len(features_selected)))
        print('\n Features selected:')
        print(features_selected)
        print('\n Features removed:')
        print(features_removed)
        print(' ')
        print(vif_table[features_selected].sort_values())

    def unnecessary_run_vif(self, num_columns, selected_features_subset):
        """Check if the number of subsets is smaller than optimal subset already computed to avoid computational complexity

        Parameters
        ----------
        num_columns : int
            Number of features that it will be used to calculate the vif scores
        selected_features_subset : dict
            Dictionary that holds computed scores from acepted features subsets

        Returns
        -------
        bool
            Response to the question 'Is it unnecessary to run this vif score?'
        """

        if 'best_num_columns' in selected_features_subset and \
            num_columns < selected_features_subset['best_num_columns']:
            return True
        else:
            return False

    def update_vif_history(self, X, vif_table, selected_features_subset):
        """Update the VIF scores history 

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with features to be analyze
        vif_table : pandas.core.frame.Series
            Vif scores for each feature selected
        selected_features_subset : dict
            Dictionary that holds computed scores from acepted features subsets

        Returns
        -------
        selected_features_subset : dict
            Dictionary that holds computed scores from acepted features subsets
        """

        vif_num_columns = len(X.columns.values)
        sum_vif = vif_table.sum()

        if 'best_num_columns' in selected_features_subset and \
            'best_sum_vif' in selected_features_subset:

            if vif_num_columns < selected_features_subset['best_num_columns'] or \
                (vif_num_columns == selected_features_subset['best_num_columns'] and \
                 sum_vif < selected_features_subset['best_sum_vif']):
                return selected_features_subset
            else:
                selected_features_subset['best_num_columns'] = vif_num_columns
                selected_features_subset['best_sum_vif'] = sum_vif
        else:
            selected_features_subset['best_num_columns'] = vif_num_columns
            selected_features_subset['best_sum_vif'] = sum_vif


        selected_features_subset[tuple(X.columns.values)] = {
            'num_columns': vif_num_columns,
            'vif_table': vif_table,
            'sum': sum_vif
        }

        return selected_features_subset

    def select_best_features_through_vif(self, X, threshold, search_limit=3, selected_features_subset={}):
        """Recursive function that aims to find the best features subset that reduces multicollinearity

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with features to be analyze
        threshold : int
            Threshold that uses VIF scores to determine multicollinearity on independent features (default is
        10)
        search_limit : int
            Search limit that reduce computational time to find the best features subset that
            is under the threshold. For the faster run set to 1.
        selected_features_subset : dict
            Dictionary that holds computed scores from acepted features subsets

        Returns
        -------
        selected_features_subset : dict
            Dictionary that holds computed scores from acepted features subsets
        """
        # Checks if the execution is worth running
        if self.unnecessary_run_vif(len(X.columns.values), selected_features_subset):
            return {}
        
        # Calculate VIF scores
        vif_table = pd.Series([variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])], index=X.columns)
        
        # Sort to remove the biggest VIFs
        vif_table.sort_values(ascending=False, inplace=True)

        # Presume that the actual selection it's good enough (doesn't pass the threshold)
        good_selection = True
        
        n_iter = 0
        for item in vif_table.iteritems():

            # Doesnt consider the constant value that is primordial to calculate VIF
            if item[0] in ['const', 'constant']: continue

            if item[1] >= threshold:

                # Excluding the bad feature from the next function call
                preselected_columns = X.columns.difference([item[0]])

                # Check if it doesnt has passed through this path before
                if tuple(preselected_columns) in selected_features_subset:
                    good_selection = False
                    continue
                
                # Limit runs to decrease computational complexity
                if n_iter > search_limit:
                    break

                n_iter = n_iter + 1
            
                recursion_response = self.select_best_features_through_vif(
                    X[preselected_columns], threshold, search_limit, selected_features_subset)
                selected_features_subset.update(recursion_response)

                good_selection = False


        if good_selection:
            selected_features_subset = self.update_vif_history(X, vif_table, selected_features_subset)

        return selected_features_subset

    def null_values_warning(self, X):
        if X.isnull().any().any():
            raise Exception('The dataframe must not have null/nan values')
        return True

    def remove_multicollinearity(self, X, threshold=10, search_limit=3):
        """Detect and remove multicollinearity from independent features

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Dataframe with features to be analyze
        threshold : int
            Threshold used VIF scores to determine multicollinearity on features (default is
        10)
        search_limit : int
            Search limit that reduce computational time to find the best features subset that
            is under the threshold. For the faster run just set 1.

        Returns
        -------
        good_features : list
            List of independent features selected to decrease multicollinearity
        bad_features : list
            List of independent features that should be removed to decrease multicollinearity
        vif_table : pandas.core.frame.Series
            Vif scores for each feature selected
        """

        X = add_constant(X)
        best_features = self.select_best_features_through_vif(X, threshold, search_limit, {})

        # excluded the unitary indexes
        max_num_columns = best_features['best_num_columns']
        min_sum_value = best_features['best_sum_vif']
        del best_features['best_num_columns']
        del best_features['best_sum_vif']

        best_features = pd.DataFrame([*best_features.values()])

        # Filtering the best run that maximizes the number of features
        # and minimizes the VIF sum
        vif_table = best_features[((best_features['num_columns'] == max_num_columns) & \
                                   (best_features['sum'] == min_sum_value))]['vif_table'].iloc[0]

        # Excluding constant used to calculate the VIF correctly
        good_features = list(set(vif_table.index.values) - set(['const']))
        bad_features  = list(set(X.columns) - set(good_features))
        bad_features  = list(set(bad_features) - set(['const']))

        self.print_info(vif_table, len(X.columns)-1, good_features, bad_features)

        del best_features, max_num_columns, min_sum_value
        
        vif_table = vif_table[good_features]

        return good_features, bad_features, vif_table

    def calculate_mi(self, x, y, m):
        m.compute_score(x, y)
        return m.mic()

    def remove_low_mutual_information(self, X, y, splits=5):
        try:
            _ = self.null_values_warning(X)

            m = MINE()
            np.random.seed(666)

            # Drop nan values
            temp = X.join(y)

            # Get the respective columns
            all_columns = X.columns
            target_column = list(set(temp.columns) - set(all_columns))

            X = temp[all_columns]
            y = temp[target_column]

            #print(len(X))
            #print(len(y))
            mi = None
            X['rnd_unif'] = np.random.uniform(size = len(X))

            if splits > 1:
                kf = KFold(n_splits=splits)
                for k_index, _ in kf.split(X):
                    #print(k_index)
                    x_k = X.iloc[k_index]
                    y_k = y.iloc[k_index]

                    #x_k.loc[k_index,'rnd_unif'] = np.random.uniform(size = len(x_k))

                    x_k['rnd_unif'] = np.random.uniform(size = len(x_k))
                    '''
                    random = pd.DataFrame({
                        'rnd_unif': np.random.uniform(size = len(x_k)),
                        'rnd_norm': np.random.randn(len(x_k))
                    })

                    x_k = pd.concat([x_k.reset_index(), random], axis=1)
                    '''
                    if isinstance(mi, pd.DataFrame):

                        mi = mi + pd.Series(x_k.columns.values) \
                            .apply(lambda column: calculate_mi(x_k[column], y_k.squeeze(), m))
                    else:

                        mi = pd.Series(x_k.columns.values) \
                            .apply(lambda column: self.calculate_mi(x_k[column], y_k.squeeze(), m))

                mi = mi / splits

            else:

                mi = pd.Series(X.columns.values) \
                        .apply(lambda column: self.calculate_mi(X[column], y.squeeze(), m))




            # Organize the results into a Dataframe
            df = pd.DataFrame(
                {'columns':X.columns.values, \
                 'values':mi}
            ).sort_values(ascending = False, by = 'values')

            random_value = df[df['columns'] == 'rnd_unif']['values'].values[0]
            best_features = df[df['values'] > random_value]['columns'].tolist()

            self.plot_horizontal(df, 'Maximum Likelihood with the target Y')
            return best_features, df

        except Exception as e:
            print(e)

    def importance_feature(self, X, y, method, target):
        '''
        X: dataset de features
        y: variavel target
        method: string indicando metodo: 'extra_tree' ou 'random_forest'. Extramely Random Trees são mais rapidas de serem
        treinadas do que Random forest.
        -------------------------------------------
        return: dicionarios de feature importance e feature permutation {importance : feature} em ordem crescente
        '''
        X['rnd_unif'] = np.random.uniform(size = len(X))    

        if method == 'random_forest':
            if target == 'regressor':
                model = RandomForestRegressor(n_jobs=-2)
            elif target == 'classifier':
                model = RandomForestClassifier(n_jobs=-2)
            else:
                print('Especifique o tipo do modelo: "regressor" ou "classifier" ')
                return False, False

            print('Treinando modelo...')
            model.fit(X, y.squeeze())

            features = X.columns.tolist()
            importances = model.feature_importances_
            indices = np.argsort(importances)

            fi = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features))
            print('Modelo treinado!')

        elif method == 'extra_tree':

            if target == 'regressor':
                model = ExtraTreesRegressor(n_estimators=500, bootstrap=True, n_jobs=-2)
            elif target == 'classifier':
                model = ExtraTreesClassifier(n_estimators=500, bootstrap=True, n_jobs=-2)
            else:
                print('Especifique o tipo do modelo: "regressor" ou "classifier" ')
                return False, False

            print('Treinando modelo...')
            model = model.fit(X, y.squeeze())

            features = X.columns.tolist()
            importances = model.feature_importances_
            indices = np.argsort(importances)

            fi = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features))
            print('Modelo treinado!')

        else:
            print("Nenhum metodo de analise foi escolhido. Escolha entre 'decision_tree' ou 'random_forest'.")
            return False, False

        print('\nExecutando feature permutation...')
        fperm = permutation_importance(model, X, y, random_state=42, n_repeats=3)
        sorted_idx = fperm.importances_mean.argsort()
        fp = sorted(zip(map(lambda x: round(x, 4), fperm.importances_mean), features))
        print('Feature permutation executada! ')

        plot_height = int(len(features)/10)+20
        fig, ax = plt.subplots(1, 2, figsize=(12,plot_height))


        #fig.tight_layout(w_pad=5)
        ax[0].set_title('Feature Importances')
        ax[0].barh(range(len(indices)), importances[indices], tick_label=[features[i] for i in indices])

        ax[1].boxplot(fperm.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
        ax[1].set_title("Permutation Importances")

        fi = pd.DataFrame(fi, columns=['values', 'columns'])
        fp = pd.DataFrame(fp, columns=['values', 'columns'])

        random_value = fp[fp['columns'] == 'rnd_unif']['values'].values[0]
        best_features = fp[fp['values'] > random_value]['columns'].tolist()

        X.drop(['rnd_unif'], axis=1, inplace=True)
        return best_features, fi, fp

    ####################################################################################################
    ################################## CONTINUAS PARA CATEGORICAS ######################################
    ####################################################################################################
    #Heatmap de correlação com size
    # Step 1 - Make a scatter plot with square markers, set column names as labels
    def plot_correlation_heatmap(self, X):
        corr = X.corr()
        corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
        corr.columns = ['x', 'y', 'value']

        x = corr['x']
        y = corr['y']
        size = corr['value'].abs()
        mod = corr['value']

        tamanho = (len(x.unique())/2)
        fig, ax = plt.subplots(1,1,figsize=(tamanho,tamanho))
        fig.suptitle('Correlation Heatmap', fontsize=16)

        # Mapping from column names to integer coordinates
        x_labels = [v for v in sorted(x.unique())]
        y_labels = [v for v in sorted(y.unique())]
        x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
        y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

        size_scale = 500
        ax.scatter(
            x=x.map(x_to_num), # Use mapping for x
            y=y.map(y_to_num), # Use mapping for y
            s=size * size_scale, # Vector of square sizes, proportional to size parameter
            marker='s' # Use square as scatterplot marker
        )

        # Show column labels on the axes
        ax.set_xticks([x_to_num[v] for v in x_labels])
        ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
        ax.set_yticks([y_to_num[v] for v in y_labels])
        ax.set_yticklabels(y_labels)

        #step 2
        ax.grid(False, 'major')
        ax.grid(True, 'minor')
        ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
        ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

        #step 3
        ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
        ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

        #step 4
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
        color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

        def value_to_color(val):
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

        ax.scatter(
            x=x.map(x_to_num),
            y=y.map(y_to_num),
            s=size * size_scale,
            c= mod.apply(value_to_color),#color.apply(value_to_color), # Vector of square color values, mapped to color palette
            marker='s'
        )
        plt.show()

    REPLACE = 'replace'
    DROP = 'drop'
    DROP_SAMPLES = 'drop_samples'
    DROP_FEATURES = 'drop_features'
    SKIP = 'skip'
    DEFAULT_REPLACE_VALUE = 0.0
    def calculate_correlation_ratio(self, categories, measurements):
        """
        Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
        Answers the question - given a continuous value of a measurement, is it possible to know which category is it
        associated with?
        Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
        a category can be determined with absolute certainty.
        Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
        **Returns:** float in the range of [0,1]
        Parameters
        ----------
        categories : list / NumPy ndarray / Pandas Series
            A sequence of categorical measurements
        measurements : list / NumPy ndarray / Pandas Series
            A sequence of continuous measurements
        nan_strategy : string, default = 'replace'
            How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
            to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
        nan_replace_value : any, default = 0.0
            The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.
        """
        categories = convert(categories, 'array')
        measurements = convert(measurements, 'array')
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)

        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))

        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator/denominator)
        return eta

    def corr_ratio(self, X, y):
        """
        X: feature dataframe (continuous)
        y: target variable (category)
        ---------
        return: dictionary and dataframe with the correlation ratio of each variable
        """
        keys = []
        values = []

        for i in X.columns:
            keys.append(i)
            values.append(self.calculate_correlation_ratio(y, X[i]))

        dict_cr = dict(zip(keys, values))
        dict_cr = {k: v for k, v in sorted(dict_cr.items(), key=lambda item: item[1])}
        df_cr = pd.DataFrame(list(dict_cr.items()), columns=['feature', 'corr_ratio'])

        # Get only 90% of the best features - Need to be improved
        num_features = int((len(df_cr)/100)*90)
        df_cr = df_cr.sort_values(by = 'corr_ratio', ascending=False)
        best_features = df_cr[:num_features]['feature'].tolist()

        return best_features, df_cr

    def select_best_features(self, X, y, objective, continous_features=[], category_features=[]):
    
        assert isinstance(X, pd.DataFrame), 'X não é um DataFrame!'
        assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), 'y não é uma Series ou DataFrame!'
        assert objective == 'classification' or objective == 'regression', "objective deve ser 'classification' ou 'regression'"
        continuos_features_selected = []
        category_best_features = []

        X, _ = self.remove_constant_features(X)
        X, _ = self.remove_duplicated_features(X)

        if len(continous_features) > 0:
            X = X[continous_features]

            X, _ = self.remove_quasi_constant_features(X)
            good, features_rm, vif_table = self.remove_multicollinearity(X, threshold=10)
            features_rlmi, mi_table = self.remove_low_mutual_information(X[good], y, splits=5)
            
            if objective == 'classification':
                
                features_cr, cr_table = self.corr_ratio(X, y)
                features_if, fi_table, fp_table = self.importance_feature(X, y, method='extra_tree', target='classifier')
                
                continuos_features_selected = list(set(features_rqc).intersection(set(features_rm), \
                            set(features_rlmi), set(features_cr), set(features_if)))
                
            elif objective == 'regression':
                
                features_if, fi_table, fp_table = self.importance_feature(X, y, method='extra_tree', target='regressor')
                
                continuos_features_selected = list(set(features_rlv).intersection(set(features_rm), \
                            set(features_rlmi), set(features_if)))

            self.heatmap(X)

        if len(category_features) > 0:
            df = X[category_features]
            
            good_features, _, _, _ = self.remove_high_theilsu(X, category_features, threshold=0.95)
            
            if objective == 'classification':
                theilu = self.theilsu_y(X, y)
            
            drops = list(set(drops).union(set(zero_var_features)))

            category_best_features = list(set(X[category_features].columns) - set(drops))
        
        all_selected_features = list(set(continuos_features_selected).union(set(category_best_features)))

        return all_selected_features, continuos_features_selected, category_best_features