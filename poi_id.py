#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import warnings
from datetime import datetime


sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def analysis_features_completeness(df_af):
    '''
    Function:    analysis_features_completeness
    Inputs:      df
    Outputs:     completeness - dictionary contains key and percentage complete
    Description: 
        This function aims to provide insights on the completeness of all the 
        records within the dataset.
    '''
    
    # Create dict to hold the feature completeness results
    completeness = dict()
    
    # Determine the dataframe length
    df_size = len(df_af)
    
    # Using dataframe functions count rows with values, excludes np.Nan
    df_count = df_af.count()
    
    # For each column in dataframe calculate the percentage complete and add to the dictionary
    for c in df_af:
        completeness[c] = (df_count[c] / df_size)*100
        #if debug: print('Feature %s is %.2f%% complete' % (c, completeness[c]))
        if debug: print ('%s; %.2f%%' % (c, completeness[c]))   
        
    # Plot the results

    completeness = {k: v for k, v in sorted(completeness.items(), key=lambda item: item[1])}

    x = range(len(completeness))
    y = (completeness.values())
    
    plt.bar(x ,y)
    plt.title('Percentage of Complete Features')
    plt.xlabel('Feature')
    plt.ylabel('Percentage Complete (%)')
    plt.xticks(x, list(completeness.keys()), rotation='vertical')
    plt.show
    return completeness

def features_plot(fp_df, feature1, feature2):
    '''
    Function:    features_plot
    Inputs:      fp_df - dataframe of data to plot
                 feature1 - string, name of feature to plot
                 feature2 - string, name of feature to plot
    Outputs:     none
    Description: 
        This function plots the features of interest grouped by whether or not
        the values belong to a POI.
    '''
        
    # Group the dataframe by poi
    groups = fp_df.groupby('poi')
    
    # Create a fig and axis to plot the data on
    fig, ax = plt.subplots()
    
    # Iterate through the POI true and POI false groups to plot the requested features.
    for name, group in groups:
        if name:
            lblName = 'POI'
            c = 'r'
            m = 'x'
        else:
            lblName = 'Not POI'
            c = 'g'
            m = 'o'
        ax.plot(group[feature1], group[feature2], marker=m, linestyle='', color=c, label=lblName)
    
    # Update the plot with legend and titles
    ax.legend()
    plt.title('Plot of %s vs %s grouped by POI' % (feature1, feature2))
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    
def feature_selection(df, n):
    '''
    Function:    feature_selection
    Inputs:      df - dataframe of the dataset
                 n - number of features to select
    Outputs:     features_list - top features according to k highest score
                 df - return of dataframe only contianing the top features
    Description: 
        This function aims to filter the dataset to only contain the top features
        based on the highest k scores.
    '''
    from sklearn.feature_selection import SelectKBest, chi2
    
    # Drop any records that contain NaNs
    df = df.dropna()
    
    # Training Input Features
    x = df.drop(["poi"], axis = 1)
    
    # Target Values
    y = df["poi"]
    
    # Transform Array
    X_new = pd.DataFrame(SelectKBest(chi2, n).fit_transform(x, y))
    
    # Get the new list of features
    features_list = ["poi"]
    for i in range(len(X_new.columns)):
        for c in df:
            if X_new.iloc[:,i].tolist() == df[c].tolist():
                features_list.append(c)   
                
    # Filter the dataframe to only the important features
    df = df[features_list]
    
    return features_list, df
    

def run_nb_classifier(x_train, y_train, x_test, y_test):
    '''
    Function:    run_nb_classifier
    Inputs:      x_train - feature training dataset
                 y_train - label training dataset
                 x_test - feature test dataset
                 y_test - label test dataset
    Outputs:     pred - predictions made by algorithm on test data
                 acc - accuracy metric of the algorithm
                 rec - recall metric of the algorithm
                 prec - prescision metric of the algorithm
                 clf - model of the algorithm
    Description: 
        This function aims to train and predict the dataset using the Naive Bayes
        classifier
    '''
    from sklearn.naive_bayes import GaussianNB
    
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred)
    
    return pred, acc, rec, prec, clf

def run_svm_classifier(x_train, y_train, x_test, y_test, data):
    '''
    Function:    run_svm_classifier
    Inputs:      x_train - feature training dataset
                 y_train - label training dataset
                 x_test - feature test dataset
                 y_test - label test dataset
                 data - dataset
    Outputs:     pred - predictions made by algorithm on test data
                 acc - accuracy metric of the algorithm
                 rec - recall metric of the algorithm
                 prec - prescision metric of the algorithm
                 clf - model of the algorithm
    Description: 
        This function aims to train and predict the dataset using the Support Vector Machine
        classifier.
        This function uses GridSearch to find the best parameters for the SVM classifier.
    '''
    from sklearn import svm
    
    if tune_required:
        from sklearn.model_selection import GridSearchCV
        cw = {0:1, 1:5} # Manually Tuned
        GSclf = GridSearchCV(svm.SVC(gamma = "auto", class_weight = cw), {"C":[1,2,3,4,5,10,20], "kernel":["linear", "poly", "rbf"], "degree":[2,3,4]}, iid = False, cv = 5)
        GSclf.fit(data.drop(["poi"], axis = 1), data["poi"])
        
        clf = svm.SVC(GSclf.best_params_["C"], GSclf.best_params_["kernel"], GSclf.best_params_["degree"], gamma = "auto", class_weight = cw)
    else:
        clf = svm.SVC()
        
    clf.fit(x_train, y_train)
    
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred)
    
    return pred, acc, rec, prec, clf

def run_tree_classifier(x_train, y_train, x_test, y_test, data):
    '''
    Function:    run_tree_classifier
    Inputs:      x_train - feature training dataset
                 y_train - label training dataset
                 x_test - feature test dataset
                 y_test - label test dataset
                 data - dataset
    Outputs:     pred - predictions made by algorithm on test data
                 acc - accuracy metric of the algorithm
                 rec - recall metric of the algorithm
                 prec - prescision metric of the algorithm
                 clf - model of the algorithm
    Description: 
        This function aims to train and predict the dataset using the Decision Tree
        classifier.
        This function uses GridSearch to find the best parameters for the DT classifier.
    '''
    from sklearn import tree
    
    if tune_required:
        from sklearn.model_selection import GridSearchCV
        GSclf = GridSearchCV(tree.DecisionTreeClassifier(), {"min_samples_split":range(2, 25)}, iid = False, cv = 5)
        GSclf.fit(data.drop(["poi"], axis = 1), data["poi"])
    
        clf = tree.DecisionTreeClassifier(min_samples_split=GSclf.best_params_["min_samples_split"])
    else:
        clf = tree.DecisionTreeClassifier()
        
    clf.fit(x_train, y_train)
    
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred)
    
    return pred, acc, rec, prec, clf

def main():
    
    print("")
    print("")
    print("********************************************************************")
    print("Running poi.py -", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("********************************************************************")
    print("")
    print("")
    
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi','salary'] # You will need to use more features
    
    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "rb") as data_file:
        data_dict = pickle.load(data_file)
        
    '''
    Convert the data dictionary to a dataframe and replace String NaN's with numpy NaN's.
    Also change booleans to int 1 or 0.
    '''
    df = pd.DataFrame.from_dict(data_dict, orient='index')    
    
    df = df.replace(["NaN", True, False], [np.NaN, 1, 0])
    df['poi'] = df.apply(lambda row: row['poi'] == True, axis = 1).astype(int)
    
    '''
    Function to analysis the completeness of each feature within the dataset.
    '''
    count = analysis_features_completeness(df)
 
    '''
    From the feature completeness i decided that any feature that is less than 50% complete
    would not be useful for detecting poi's. Filter added for keys already exisiting
    in feature_list as poi and salary already manually added. A further filter for features 
    email and other was added as this infomation won't help detecting a poi.
    '''
    excluded_features = ['email_address', 'other']
    for feature in count:
        if count[feature] > 50.0 and feature not in features_list and \
                feature not in excluded_features:
            features_list.append(feature)
    if debug: print("Selected features: ", features_list)
    
    '''
    At this point i decided to remove the features that were not going to be used going
    forward.    
    '''    
    df = df[features_list]
    
    print("")
    print("********************************************************************")
    print("")
    
        
    ### Task 2: Remove outliers
    features_plot(df, 'salary', 'bonus')
    
    '''
    This plot shows that there is one record that has extermely high bonus and salary.
    '''
    name = df[df['salary'] == df["salary"].max()].index[0]
    if debug: print('%s has the max salary and bonus of: %.0f %.0f' % (name, df["salary"].max(), df["bonus"].max()))
    
    '''
    This showed that the outlier was the TOTAL record. This record is not useful for detecting
    a POI and was subsequentially removed.
    '''
    
    if debug: print(name, "has been removed from the dataset")
    df = df.drop(name) 
    
    features_plot(df, 'salary', 'bonus')
    name = df[df['salary'] == df["salary"].max()].index[0]
    if debug: print('%s has the max salary and bonus of: %.0f %.0f' % (name, df["salary"].max(), df["bonus"].max()))

    '''
    Reassessing the plot and dataframe infomation provided a more reasonable result.
    However from the visulation there are a number of outlying points, therefore I decided
    to scale both the salary and bonus to remove the top and bottom 10% of records that 
    were not POIs.
    ''' 
    percentage_to_remove = 0.1
    
    # Filtering the df for the features of interest to identify outliers.
    outlier_features = ['poi', 'salary', 'bonus']
    df_outlier = df[outlier_features]
    
    # Setup a MinMaxScaler in order to easily indentify the top & bottom 10%
    scalar = pp.MinMaxScaler()
    # Whilst using this scalar i found that it failed with numpy NaNs, 
    # so i replaced them with 0
    df_outlier = df_outlier.replace(np.NaN, 0)
    # Iteratre through the features to scale
    for feature in outlier_features:
        # Skip the poi feature as the values are only ever 1 or 0, so no point scaling
        if feature == 'poi': continue
        # Reshape the data in order to fit the scalar transform function
        data = df_outlier[feature].values.reshape(-1, 1)
        # Scale the data using the fit_transform function
        scaled_data = scalar.fit_transform(data)
        # Replace the df data with the new scaled values
        df_outlier[feature] = scaled_data
          
    # Setup a list to hold the outlier names
    outlier_names = []
    
    # Iterate over each row in the df
    for name, row in df_outlier.iterrows():
        # If not a poi and in the top or bottom 10% add them to the outlier list.
        if row['poi'] == 0 and \
              (row['salary'] < percentage_to_remove or row['salary']  > 1-percentage_to_remove) and \
              (row['bonus'] < percentage_to_remove or row['bonus']  > 1-percentage_to_remove):
            outlier_names.append(name)
            
    if debug: print('Number of outliers  identified: %i.' % len(outlier_names))
    if debug: print(outlier_names)
    
    '''
    This method identified 58 outliers in the top or bottom 10% of the salary and bonus feature that
    weren't POIs, however when reviewing the list of outliers they looked like they could be real 
    people, with the exception of 'THE TRAVEL AGENCY IN THE PARK'. Due to small number of records 
    in the first place i decided to only remove the travel agency.
    '''
    # Remove outliers from the orignal df
    df = df.drop('THE TRAVEL AGENCY IN THE PARK')  
    # Replot the graph
    features_plot(df, 'salary', 'bonus')
    
    '''
    2 outliers were removed from the dataset, leaving 144 records.
    '''
    
    print("")
    print("********************************************************************")
    print("")
    
    
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    '''
    To, from and shared with POI ratio would be more useful that the number of emails sent and recieved 
    from a POI. As this takes into account the total number of emails sent/recieved.
    
    I also decided that to remove the emails sent/recieved at this point as i don't expect to use them
    going forward.
    
    '''    
    df['ratio_from'] = df['from_this_person_to_poi'] / df['from_messages']
    df['ratio_to'] = df['from_poi_to_this_person'] / df['to_messages']
    df['ratio_shared'] = df['shared_receipt_with_poi'] / (df['to_messages'] +  df['from_messages'])
    
    features_list.append('ratio_from')
    features_list.append('ratio_to')
    features_list.append('ratio_shared')
    features_list.remove('from_this_person_to_poi')
    features_list.remove('from_messages')
    features_list.remove('from_poi_to_this_person')
    features_list.remove('to_messages')
    features_list.remove('shared_receipt_with_poi')
    
    
    '''
    At this stage i decieded to scale all the remaining features with the exclusion of the
    poi and calculated ratios, as these values are already between 0 and 1
    '''
    features_excluded_from_scale = ['poi', 'ratio_from', 'ratio_to', 'ratio_shared']
    
    # Filter the df to only contain the features remaining in the feature list
    df = df[features_list]
    # As before numpy NaN's cause an error whilst scaling so replaced with 0
    df = df.replace(np.NaN, 0)
    # Iterate over each feature in the df
    for feature in df:
        # Skip feature if in the exclusion list
        if feature in features_excluded_from_scale: continue
        # Reshape the data in order to fit the scalar transform function
        data = df[feature].values.reshape(-1, 1)
        # Scale the data using the fit_transform function
        scaled_data = scalar.fit_transform(data)
         # Replace the df data with the new scaled values
        df[feature] = scaled_data
    
    '''
    Function to find the most usefuel features, further details provide in the function
    description
    '''
    if debug: print('Complete features list:', " ".join(features_list), '\n')
    no_features = 5
    no_features = [2, 4, 5, 6, 8]
    mean_score = []
    
    if debug: print('\nFeatures Selection Results:')
    if debug: print('NoFeatures;Type;Acc;Rec;Prec;Score')
    
    for k in range(len(no_features)):
        score = np.zeros([1,3])
        
        fl, features_df = feature_selection(df[features_list], no_features[k])
        
        labels = features_df['poi']
        features = features_df.drop(["poi"], axis = 1)
        
        x_train, x_test, y_train, y_test = train_test_split(features, labels, \
                                train_size=0.6, random_state=38)
        
        pred, acc, rec, prec, clf = run_nb_classifier(x_train, y_train, x_test, y_test)
        score[0,0] = acc + rec + prec
        if debug: print ('%i;NB;%.2f;%.2f;%.2f;%.2f' %(no_features[k], acc, rec, prec,  score[0,0]))
                
        pred, acc, rec, prec, clf = run_svm_classifier(x_train, y_train, x_test, y_test, features_df)
        score[0,1] = acc + rec + prec
        if debug: print ('%i;SVM;%.2f;%.2f;%.2f;%.2f' %(no_features[k], acc, rec, prec,  score[0,1]))
        
        pred, acc, rec, prec, clf = run_tree_classifier(x_train, y_train, x_test, y_test, features_df)
        score[0,2] = acc + rec + prec
        if debug: print ('%i;DT;%.2f;%.2f;%.2f;%.2f' %(no_features[k], acc, rec, prec,  score[0,2]))
        
        mean_score.append(score.mean())
    
    max_index = np.argmax(mean_score, axis=0)
    if debug: print('Highest scoring feature list: ', no_features[max_index])
    
    features_list, features_df = feature_selection(df[features_list], no_features[max_index])
    print('Final features list:', " ".join(features_list), '\n')
    '''
    Before implementing the scaling function the feature selection function selected 
    the total_payments, total_stock_value & exercised_stock_options features as the 
    most useful features to use to determine who is a poi.
    
    However, after scaling the feature selection function selected the total_stock_value, 
    exercised_stock_options & ratio_from features as the most useful features to use to 
    determine who is a poi.
    '''
    
    print("")
    print("********************************************************************")
    print("")
    
    
    '''
    Convert the features_df back to a dict for the template code
    '''
    data_dict = features_df.to_dict(orient = 'index')
    my_dataset = data_dict
    
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    
    '''
    The started code targetFeatureSplit has been removed as i opted to use a df for
    majority of the analysis rather than a dict.
    '''
    #labels, features = targetFeatureSplit(data)
        
    labels = features_df['poi']
    features = features_df.drop(["poi"], axis = 1)
        
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    # Provided to give you a starting point. Try a variety of classifiers.
    results = []
    raw_results = []
 
    print("")
    print("")
    print("********************************************************************")
    print("Results:")
    
       
    '''
    Use the StratifiedShuffleSplit function to ensure both train and test dataset have
    enough records with POI to accurately measure the performance metrics.
    '''
    
    sss = StratifiedShuffleSplit(n_splits=20, test_size=0.4, random_state=12)

    print('Index;Type;Acc;Rec;Prec;Score')
    i = 0
    for train_index, test_index in sss.split(features, labels):
        
        x_train = features.iloc[train_index]
        y_train = labels.iloc[train_index]
        x_test = features.iloc[test_index]
        y_test = labels.iloc[test_index]

        pred, acc, rec, prec, clf = run_nb_classifier(x_train, y_train, x_test, y_test)
        value = {'type':'NB', 'index':i, 'predicitions':pred, 'accuracy':acc, \
                 'recall':rec, 'precision':prec, 'clf':clf}
        raw_results.append(value)
        if rec > 0.4 and prec > 0.4:
            results.append(value)
        
        print('%.0f;NB;%.4f;%.4f;%.4f;%.4f' % (i, acc, rec, prec, acc + rec + prec))
        
        pred, acc, rec, prec, clf = run_svm_classifier(x_train, y_train, x_test, y_test, features_df)
        value = {'type':'SVM', 'index':i, 'predicitions':pred, 'accuracy':acc, \
                 'recall':rec, 'precision':prec, 'clf':clf}
        raw_results.append(value)
        if rec > 0.4 and prec > 0.4:
            results.append(value)
        
        print('%.0f;SVM;%.4f;%.4f;%.4f;%.4f' % (i, acc, rec, prec, acc + rec + prec))
        
        pred, acc, rec, prec, clf = run_tree_classifier(x_train, y_train, x_test, y_test, features_df)
        value = {'type':'DT', 'index':i, 'predicitions':pred, 'accuracy':acc, \
                 'recall':rec, 'precision':prec, 'clf':clf}
        raw_results.append(value)
        if rec > 0.4 and prec > 0.4:
            results.append(value)
        
        print('%.0f;DT;%.4f;%.4f;%.4f;%.4f' % (i, acc, rec, prec, acc + rec + prec))
        i += 1
        
    print("********************************************************************")
    print("")
    print("")
    
    df_results = pd.DataFrame(raw_results, columns=['type','index','predicitions','accuracy','recall','precision','clf'])
    df_results['composite'] = df_results['accuracy'] + df_results['recall'] + df_results['precision']
    
    id_best = df_results['composite'].idxmax()
    print('The best results were achieved using %s, in index: %i. Achieving an Accuracy of %.1f%%, Recall of %.1f%% & Precision of %.1f%%.'\
          % (df_results['type'].get(id_best), df_results['index'].get(id_best), \
             df_results['accuracy'].get(id_best) * 100, df_results['recall'].get(id_best) * 100, \
             df_results['precision'].get(id_best) * 100))

    print("")
    print("")
    
    clf = df_results['clf'].get(id_best)
    
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    
    '''
    Tuning done within task 4 segment, and within the individual algorithm functions
    '''
    
    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    
    dump_classifier_and_data(clf, my_dataset, features_list)
    
    return df_results

debug = True
tune_required = True

if __name__ == "__main__":
    df = main()
    
    