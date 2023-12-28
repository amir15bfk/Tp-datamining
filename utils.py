import math
import matplotlib.pyplot as plt
import random
from collections import Counter

# The Evaluator class is a Python class that calculates various evaluation metrics such as accuracy,
# recall, precision, F1 score, specificity, and generates a confusion matrix for a classification
# model.
class Evaluator:
    def __init__(self,real,pred,classes):
        """
        The function initializes a confusion matrix and populates it based on the real and predicted
        values.
        
        :param real: The `real` parameter is a list that represents the true labels or classes of the
        data. It is expected to be a list of integers, where each integer represents a class label. The
        `real` list is sliced using `[1:]` to exclude the first element
        :param pred: The `pred` parameter is a list that represents the predicted classes for a set of
        data points. Each element in the list corresponds to the predicted class for a specific data
        point
        :param classes: The `classes` parameter represents the number of classes or categories in your
        data. In this case, it seems that there are 3 classes
        """
        self.real = real[1:]
        self.pred = pred
        self.classes = classes
        self.matrix = [[0]*3,[0]*3,[0]*3]
        for i in range(len(self.real)):
            self.matrix[self.real[i]][self.pred[i]]+=1
    def accuracy(self):
        """
        The above function calculates the accuracy of a classification model by summing the diagonal
        elements of a confusion matrix and dividing it by the total number of samples.
        :return: the accuracy of a classification model. It calculates the number of true positive
        predictions (diagonal elements of the confusion matrix) and divides it by the total number of
        instances in the dataset.
        """
        true = sum([self.matrix[i][i] for i in range(len(self.matrix))])
        return true/len(self.real)
    def recall(self):
        """
        The above function calculates the recall for each class based on the values in a confusion
        matrix.
        :return: a list of recall values for each class in the confusion matrix.
        """
        recall = []
        for i in range(len(self.matrix)):
            TP = self.matrix[i][i]
            FN_TP = sum(self.matrix[i]) 

            recall.append((TP/FN_TP) if TP!=0 else 0)
        return recall
    def precision(self):
        """
        The function calculates the precision for each class based on the values in a confusion matrix.
        :return: a list of precision values for each class in the confusion matrix.
        """
        precision = []
        for i in range(len(self.matrix)):
            TP = self.matrix[i][i]
            FP_TP = sum([self.matrix[j][i] for j in range(len(self.matrix))]) 
            precision.append((TP/FP_TP) if TP!=0 else 0)
        return precision
    def f1_score(self):
        """
        The function calculates the F1 score using the recall and precision values.
        :return: a list of F1 scores.
        """
        recall = self.recall()
        precision = self.precision()
        f1_score = []
        for r,p in zip(recall,precision):
            f1_score.append(((2*r*p)/(r+p)) if (r+p)!=0 else 0)
        return f1_score
    def specificity(self):
        # The above code is calculating the specificity for each class in a confusion matrix. It
        # iterates over each row in the matrix and calculates the true negatives (TN) and false
        # positives (FP) for that class. It then calculates the specificity by dividing TN by the sum
        # of TN and FP, and appends it to the specificity list. If TN is 0, it appends 0 to the
        # specificity list. Finally, it returns the specificity list.
        specificity = []
        for i in range(len(self.matrix)):
            TN = sum([self.matrix[j][k] for j in range(len(self.matrix))for k in range(len(self.matrix)) if (k!=i) and (j!=1)])
            FP = sum([self.matrix[j][i] for j in range(len(self.matrix))]) -self.matrix[i][i]
            specificity.append((TN/(FP+TN)) if TN!=0 else 0)
        return specificity
    def confusion_matrix(self):
        """
        The function confusion_matrix returns the matrix attribute of the object.
        :return: The confusion matrix.
        """
        return self.matrix
    def report(self):
        """
        The function "report" generates a report containing various evaluation metrics such as recall,
        precision, f1-score, specificity, accuracy, and a confusion matrix.
        :return: a list of strings. The list contains a report, including recall, precision, f1-score,
        specificity, accuracy, and a confusion matrix.
        """
        recall = self.recall()
        precision = self.precision()
        f1_score = self.f1_score()
        specificity = self.specificity()
        accuracy = self.accuracy()
        out = []
        out.append("Report:")
        out.append("        recall     precision   f1-score     specificity")
        for i in self.classes:
            out.append(f"  {i}      { 0 if recall[i]<0.1 else ''}{recall[i]*100:2.2f}       { 0 if precision[i]<0.1 else ''}{precision[i]*100:2.2f}       { 0 if f1_score[i]<0.1 else ''}{f1_score[i]*100:2.2f}        { 0 if specificity[i]<0.1 else ''}{specificity[i]*100:2.2f}")

        out.append(f"accuracy : {accuracy:.2f}")
        out.append("Confusion Matrix : ")
        for i in self.matrix:
            out.append(str(i))
        return out

def median(column):
    """
    The function "median" calculates the median value of a given column.
    
    :param column: The parameter "column" represents a list of numbers
    :return: The median value of the given column.
    """
    sc = sorted(column)
    if len(sc)%2 != 0:
        return sc[len(column)//2]  
    else:
        return (sc[len(column)//2] +sc[len(column)//2+1] )/2

def mean(l): 
    """
    The mean function calculates the average of a list of numbers.
    
    :param l: l is a list of numbers
    :return: The mean of the given list 'l' is being returned.
    """
    mean = sum(l)/len(l)
    return mean

def mode(column):
    """
    The `mode` function takes a column of values and returns a list of the most frequently occurring
    values in the column.
    
    :param column: The parameter "column" is a list of values
    :return: a list of mode(s) from the given column.
    """
    counter = dict()
    max_count = 0
    for val in column:
        if counter.get(val):
            counter[val] += 1
        else:
            counter[val] = 1
        max_count = max(max_count, counter[val])

    modes = [val for val, count in counter.items() if count == max_count]
    return modes

def sd(column,m):
    """
    The function calculates the standard deviation of a given column of data using the formula
    sqrt(sum((val-m)**2)/(len(column)-1)), where m is the mean of the column.
    
    :param column: The "column" parameter represents a list of values. It is the column of data for
    which you want to calculate the standard deviation
    :param m: The parameter "m" represents the mean of the column
    :return: The standard deviation of the column values.
    """
    return math.sqrt(sum([(val-m)**2 for val in column])/(len(column)-1))

def Q(column):
    """
    The function Q calculates the minimum, first quartile, median, third quartile, and maximum values of
    a given column.
    
    :param column: The parameter "column" represents a list of numerical values
    :return: a list containing the minimum value, first quartile (Q1), median (Q2), third quartile (Q3),
    and maximum value of the input column.
    """
    ordlist = sorted(column)
    min_c = min(column)
    if len(ordlist)%4 != 0:
        Q1 =  ordlist[len(column)//4]  
    else:
        Q1 = (ordlist[len(column)//4] +ordlist[len(column)//4+1] )/2
    Q2= median(column)
    if len(ordlist)%4 != 0:
        Q3 =  ordlist[(3*len(column))//4]  
    else:
        Q3 = (ordlist[(3*len(column))//4] +ordlist[(3*len(column))//4+1] )/2
    max_c=max(column)

    return [min_c,Q1,Q2,Q3,max_c]

def fixWithMean(df,do_not = [] ):
    """
    The function "fixWithMean" takes a dataframe and a list of columns to exclude, and replaces any
    missing values in the specified columns with the mean value of the non-missing values in that
    column.
    
    :param df: The parameter `df` is a 2-dimensional list or a dataframe containing the data that needs
    to be fixed. Each row represents a record and each column represents a variable
    :param do_not: The `do_not` parameter is a list of column indices that should not be fixed with the
    mean. These columns will be skipped and not modified in the dataframe
    """
    for col in range(len(df[0])):
        if col not in do_not:
            l = [row[col] for row in df[1:] if row[col]!=None]
            m = mean(l)
            for row in range(1,len(df)):
                if df[row][col]==None:
                    df[row][col]=m
            
def stats(df):
    """
    The function `stats` calculates various statistics (mean, median, mode, standard deviation,
    quartiles) for each column in a given dataframe and displays boxplots for each column.
    
    :param df: The parameter `df` is a list of lists representing a data frame. Each inner list
    represents a row in the data frame, and the first inner list represents the column names. The data
    frame should have numerical values, and missing values should be represented as `None`
    """
    ls = []
    for col in range(len(df[0])):
        l = [row[col] for row in df[1:] if row[col]!=None]
        ls.append(l)
        miss_vals = len(df)-1-len(l)
        print(df[0][col] , " : mean =",round(mean(l),2),"median =",median(l),"mode =",mode(l),"standard deviation =",round(sd(l,mean(l)),2),"Q =",Q(l),"miss_vals =",miss_vals)
    size = len(df[0])-1
    fig, axes = plt.subplots(1, size, figsize=(20, 4))
    plt.subplots_adjust(wspace=0.7)
    for i in range(size):
        ax = axes[i]
        ax.boxplot(ls[i],labels=[df[0][i]]) 
    plt.show()

def histogrames(df):
    """
    The function `histogrames` takes a dataframe as input and plots histograms for each column in a 2x7
    grid.
    
    :param df: The parameter `df` is expected to be a list of lists, where each inner list represents a
    row in a dataset. The first inner list is assumed to contain the column names, and the remaining
    inner lists represent the data rows
    """
    ls = []
    for col in range(len(df[0])):
        l = [row[col] for row in df[1:] if row[col]!=None]
        ls.append(l)
        
    size = len(df[0])
    fig, axes = plt.subplots(2, 7, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.1)
    for i in range(size):
        ax = axes[i%2][i//2]
        ax.hist(ls[i],bins=30,edgecolor='black')
        ax.set_title(df[0][i])
    plt.show()

def removeAberr(df,do_not = []):
    """
    The function `removeAberr` removes aberrant values from a given dataframe, excluding specified
    columns.
    
    :param df: The parameter `df` is a 2-dimensional list representing a dataframe. Each row in the list
    represents a row in the dataframe, and each element in the row represents a value in the dataframe
    :param do_not: The `do_not` parameter is a list of column indices that should not be processed for
    removing aberrant values
    """
    for col in range(len(df[0])):
        if col not in do_not:
            l = [row[col] for row in df[1:] if row[col]!=None]
            q = Q(l)
            iqr = q[3]-q[1]
            for row in range(1,len(df)):
                if (df[row][col]>(q[3]+1.5*iqr)) or (df[row][col]<(q[1]-1.5*iqr)):
                    df[row][col]=None

def fixAbb(df,do_not = []):
    """
    The function "fixAbb" removes aberrant values from a dataframe and replaces them with the mean value
    of the column.
    
    :param df: The input dataframe that needs to be fixed
    :param do_not: The `do_not` parameter is a list of column names that should not be fixed for
    aberrant values
    """
    removeAberr(df,do_not)
    fixWithMean(df)

def remove_hozizontale(df):
    """
    The function removes duplicate rows from a given dataframe.
    
    :param df: The parameter `df` is a list of lists. The first element of the list is a header row, and
    the remaining elements are data rows. Each data row is also a list
    :return: a modified version of the input dataframe. The first element of the dataframe is kept as
    is, and the remaining rows are converted to a set of tuples, then back to a list of lists.
    """
    return [df[0]]+list(map(list, set(map(tuple, df[1:]))))

def normMinMax(l,min_new=0,max_new=1):
    """
    The function `normMinMax` takes a list `l` and normalizes its values between `min_new` and `max_new`
    using the minimum and maximum values of the original list.
    
    :param l: The parameter "l" is a list of numbers that you want to normalize
    :param min_new: The minimum value that you want to scale the data to. By default, it is set to 0,
    defaults to 0 (optional)
    :param max_new: The max_new parameter is the desired maximum value for the normalized data. It is
    used to scale the data to a specific range, defaults to 1 (optional)
    :return: The function `normMinMax` returns a tuple containing the normalized list `l`, the minimum
    value of the original list `min_old`, and the maximum value of the original list `max_old`.
    """
    min_old=min(l)
    max_old=max(l)
    for i in range(len(l)):
        l[i] = round((l[i]-min_old)/(max_old-min_old)*(max_new-min_new)+min_new,2)
    return l ,min_old,max_old

def normZ_score(l,min_new=0,max_new=1):
    """
    The function normZ_score calculates the normalized Z-score for each element in a list and returns
    the normalized list, mean, and standard deviation.
    
    :param l: The parameter "l" is a list of numbers for which you want to calculate the normalized
    z-scores
    :param min_new: The `min_new` parameter is the minimum value of the new range that you want to
    normalize the data to. By default, it is set to 0, defaults to 0 (optional)
    :param max_new: The `max_new` parameter is the maximum value that you want the normalized values to
    be scaled to. By default, it is set to 1, defaults to 1 (optional)
    :return: a list with the normalized Z-scores of the input list 'l', as well as the mean 'm' and
    standard deviation 's' of the original list.
    """
    m=mean(l)
    s=sd(l,m)
    for i in range(len(l)):
        l[i] = (l[i]-m)/s
    return l , m,s

def normHist(df,MM=True,Z=True,plot=True,do_not = []):
    """
    The `normHist` function normalizes the columns of a given dataframe using either Z-score
    normalization or Min-Max normalization, and then plots the histograms of the normalized columns.
    
    :param df: The input dataframe containing the data to be normalized
    :param MM: The parameter "MM" stands for Min-Max normalization. If set to True, it will normalize
    the data using the Min-Max normalization technique. If set to False, it will skip the Min-Max
    normalization step, defaults to True (optional)
    :param Z: The parameter "Z" in the function "normHist" determines whether or not to perform Z-score
    normalization on the data. If set to True, Z-score normalization will be performed. If set to False,
    Z-score normalization will not be performed, defaults to True (optional)
    :param plot: The "plot" parameter determines whether or not to display histograms of the normalized
    data. If set to True, histograms will be displayed. If set to False, histograms will not be
    displayed, defaults to True (optional)
    :param do_not: The `do_not` parameter is a list that allows you to specify the columns that should
    not be normalized. This means that the normalization process will be skipped for those columns
    :return: a list containing the normalized histogram data for each column in the input dataframe. It
    also returns the minimum and maximum values used for normalization (if MM=True), and the mean and
    standard deviation values used for normalization (if Z=True).
    """
    ls = []
    min_norm = []
    max_norm = []
    mean_norm = []
    std_norm = []
    for col in range(len(df[0])):
            l = [row[col] for row in df[1:] if row[col]!=None]
            if col not in do_not:
                if Z:
                    l,m,s = normZ_score(l)
                    mean_norm.append(m)
                    std_norm.append(s)
                if MM:
                    l,min_c,max_c = normMinMax(l)
                    max_norm.append(max_c)
                    min_norm.append(min_c)
                
            ls.append(l)
        
    size = len(df[0])-1
    if plot:
        fig, axes = plt.subplots(2, 7, figsize=(20, 8))
        plt.subplots_adjust(wspace=0.1)
        for i in range(size):
            ax = axes[i%2][i//2]
            ax.hist(ls[i],bins=30,edgecolor='black')
            ax.set_title(df[0][i])
        plt.show()
    return  [df[0]] + list(map(list, zip(*ls))), min_norm,max_norm,mean_norm,std_norm

def dest_manhat(v1,v2):
    """
    The function `dest_manhat` calculates the Manhattan distance between two vectors `v1` and `v2`.
    
    :param v1: The first vector, represented as a list of numbers
    :param v2: The parameter v2 is a list representing the destination coordinates
    :return: The function `dest_manhat` returns the sum of the absolute differences between
    corresponding elements of two input vectors `v1` and `v2`.
    """
    s = 0
    for i in range(len(v1)):
        s+= abs(v1[i]-v2[i])
    return s

def dest_eclude(v1,v2):
    """
    The function calculates the Euclidean distance between two vectors.
    
    :param v1: The parameter `v1` is a list representing the first vector. Each element of the list
    represents a component of the vector
    :param v2: The parameter v2 is a list of numbers
    :return: the square root of the sum of the squared differences between the elements of v1 and v2.
    """
    s = 0
    for i in range(len(v1)):
        s+= (v1[i]-v2[i])**2
    return math.sqrt(s)

def dest_minko(v1,v2,p=2):
    """
    The function `dest_minko` calculates the Minkowski distance between two vectors `v1` and `v2` using
    the specified power `p`.
    
    :param v1: v1 is a list representing the first vector
    :param v2: The parameter `v2` is a list representing the second vector
    :param p: The parameter "p" is the power to which the differences between corresponding elements of
    "v1" and "v2" are raised before summing them up. It is set to 2 by default, which means that the
    Euclidean distance is calculated. However, you can change the value of, defaults to 2 (optional)
    :return: the Minkowski distance between two vectors, v1 and v2, with an optional parameter p.
    """
    s = 0
    for i in range(len(v1)):
        s+= math.pow((v1[i]-v2[i]),p)
    return math.pow(s,1/p)

def dest_cos(v1,v2):
    """
    The function `dest_cos` calculates the cosine distance between two vectors.
    
    :param v1: A list representing the first vector
    :param v2: The parameter `v2` represents a vector
    :return: the cosine distance between two vectors.
    """
    s = 0
    sa = 0
    sb = 0
    for i in range(len(v1)):
        s+= (v1[i]*v2[i])
        sa+= (v1[i]*v1[i])
        sb+= (v2[i]*v2[i])
    return 1 - s / (math.sqrt(sa)*math.sqrt(sb))

def dest_ham(v1,v2):
    """
    The function `dest_ham` calculates the Hamming distance between two vectors `v1` and `v2`.
    
    :param v1: The first parameter, v1, is a list representing a vector
    :param v2: The parameter v2 is a list of values
    :return: the number of positions where the elements of v1 and v2 are different.
    """
    s = 0
    for i in range(len(v1)):
        s+= 1 if v1[i] != v2[i] else 0
    return s

def find_dom_in_k(v):
    """
    The function `find_dom_in_k` returns the most common element in a given list `v`.
    
    :param v: The parameter "v" is a list of values
    :return: the most common element in the input list 'v'.
    """
    return Counter(v).most_common()[0][0]

def plot_clusters(clusters,x1,x2,columns1_name,columns2_name):
    """
    The function `plot_clusters` takes in a list of clusters and two column indices, and plots the data
    points in each cluster on a scatter plot using the specified columns as x and y coordinates.
    
    :param clusters: The clusters parameter is a list of clusters. Each cluster is represented as a list
    of data points. Each data point is represented as a list or tuple of values
    :param x1: The index of the column to be plotted on the x-axis
    :param x2: The parameter `x2` represents the index of the column in the dataset that you want to use
    as the y-axis for plotting the clusters
    :param columns1_name: The name of the first column in your dataset
    :param columns2_name: The parameter "columns2_name" is the name of the second column in your
    dataset. It is used as the label for the y-axis in the plot
    :return: a figure object.
    """
    fig ,ax = plt.subplots()
    for cluster in clusters:
        ax.scatter([i[x1] for i in cluster[1:]],[i[x2] for i in cluster[1:]])
        ax.set_xlabel(columns1_name)
        ax.set_ylabel(columns2_name)
    return fig

def silhouette_coefficient(clusters,f_dist = dest_eclude):
    """
    The function calculates the silhouette coefficient for a given set of clusters using a specified
    distance function.
    
    :param clusters: The clusters parameter is a list of clusters. Each cluster is represented as a list
    of data points. The data points can be any type of object, as long as the f_dist function can
    calculate the distance between them
    :param f_dist: The parameter `f_dist` is a function that calculates the distance between two data
    points. The default value is `dest_eclude`, which is likely a typo and should be
    `euclidean_distance`. This function should take two data points as input and return the distance
    between them
    :return: the silhouette coefficient, which is a measure of how well-defined the clusters are in a
    clustering algorithm.
    """
    S = 0
    for idx_c,cluster in enumerate(clusters):
        n_lk = len(cluster)-1
        Si = 0
        for i,vi in enumerate(cluster[1:]):

            ai = 0
            for j,vj in enumerate(cluster[1:]):
                if i!=j:
                    ai+=f_dist(vi,vj)
            try:
                ai *= (1/(n_lk-1))
            except:
                ai = 0

            bi = []
            for idx_c_hat,cluster_hat in enumerate(clusters):
                if idx_c_hat!=idx_c:
                    bbi = 0
                    n_lk_hat = len(cluster_hat)-1
                    for i_hat,vi_hat in enumerate(cluster_hat[1:]):
                        bbi+=f_dist(vi,vi_hat)
                    bi.append(bbi*(1/n_lk_hat))
            try :
                bi = min(bi)
            except:
                bi=0
            Si += (bi-ai)/max(bi,ai)
        S+=((1/n_lk)*Si)
    return len(clusters)*S

def load_data_par_row(path,columns,col_type,rm_end=False):
    """
    The function `load_data_par_row` reads a CSV file line by line, converts each value to the specified
    data type, and returns the data as a list of lists.
    
    :param path: The path parameter is the file path of the data file that you want to load
    :param columns: The "columns" parameter is a list that specifies the column names for the data. Each
    element in the list represents a column name
    :param col_type: The `col_type` parameter is a list of functions that specify the data type
    conversion for each column in the data. For example, if you have a column that contains integers,
    you can specify `int` as the corresponding function in the `col_type` list. This will convert the
    values in
    :param rm_end: The "rm_end" parameter is a boolean value that determines whether or not to remove
    the end character (usually a newline character) from each line before splitting it into columns. If
    "rm_end" is set to True, the end character will be removed. If it is set to False, the, defaults to
    False (optional)
    :return: a list of lists, where each inner list represents a row of data from the file. The first
    inner list contains the column names specified in the "columns" parameter.
    """
    
    with open(path,'r') as f:
        
        start =True
        for line in f:
            if start:
                
                df= [columns]

                start =False
            else:
                if rm_end:
                    row = line[:-1].split(",")
                else:
                    row = line.split(",")
                n_row = []
                for i,c in enumerate(row):
                    try:
                        c = col_type[i](c)
                    except: 
                        c = None
                    n_row.append(c)
                df.append(n_row)
    return df

def train_test_split(X,test_per=0.2,seed=55555):
    """
    The function `train_test_split` takes a dataset `X` and splits it into training and testing sets,
    with a specified test percentage and random seed.
    
    :param X: X is the dataset that you want to split into training and testing sets. It should be a
    list of lists, where each inner list represents a data point and its corresponding features and
    labels
    :param test_per: The test_per parameter is the percentage of data that should be allocated for
    testing. It is set to 0.2 by default, which means that 20% of the data will be used for testing
    :param seed: The seed parameter is used to initialize the random number generator. It ensures that
    the random splits generated are reproducible. By setting a specific seed value, you can obtain the
    same train-test splits every time you run the function, defaults to 55555 (optional)
    :return: The function `train_test_split` returns six values: `train`, `test`, `x_train`, `x_test`,
    `y_train`, and `y_test`.
    """
    train = [X[0]]
    test = [X[0]]
    test_size = round((len(X)-1)*0.2)
    random.seed(seed)
    for i in X[1:]:
        if test_size>0:
            if random.random()>=0.5:
                test.append(i)
                test_size-=1
            else:
                train.append(i)
        else:
            train.append(i)
    x_train = [i[:-1] for i in train]
    x_test = [i[:-1] for i in test]
    y_train = [i[-1] for i in train]
    y_test = [i[-1] for i in test]
    return train ,test ,x_train,x_test,y_train,y_test

def compute_k(l):
    """
    The function computes the value of k using a formula that involves the length of a given list.
    
    :param l: The parameter "l" is a list
    :return: the value of the variable "k".
    """
    k = round(1+10/3*math.log10(len(l)))
    return k

def div_to_k_frequency(l,k=None,classes=None,replace_with_moy=False):
    """
    The function `div_to_k_frequency` takes a list of numbers and divides them into k frequency classes,
    optionally replacing the numbers with the average value of each class.
    
    :param l: The input list of numbers that you want to divide into classes
    :param k: The number of classes or categories to divide the data into. If not provided, it will be
    computed automatically based on the data
    :param classes: The `classes` parameter is a list that specifies the names of the classes or
    categories that the data will be divided into. If the `classes` parameter is not provided or its
    length is not equal to `k`, then default class names will be generated as "class_1", "class_
    :param replace_with_moy: The parameter "replace_with_moy" is a boolean flag that determines whether
    the values in the input list should be replaced with the average value of their corresponding class.
    If set to True, the function will calculate the average value for each class and replace the values
    in the input list with their respective class, defaults to False (optional)
    :return: The function `div_to_k_frequency` returns a list `out` which contains the classes to which
    each element in the input list `l` belongs.
    """
    if k==None:
        k=compute_k(l)
    if classes == None or len(classes)!=k:
        classes=["class_"+str(i+1) for i in range(k)]
    if replace_with_moy:
        classes_dict = {i:{"sum":0,"cpt":0,"moy":0}for i in classes}
    mn = min(l)
    mx = max(l)
    domain = mx-mn
    i = 0
    out = []
    while i<len(l):
        class_n = int((l[i]-mn)*k/domain)
        class_n =class_n if class_n< len(classes)else class_n - 1
        out.append(classes[class_n])
        if replace_with_moy:
            classes_dict[classes[class_n]]["sum"]+=l[i]
            classes_dict[classes[class_n]]["cpt"]+=1
        i+=1
    if replace_with_moy:
        for class_name in classes:
            classes_dict[class_name]["moy"] = round(classes_dict[class_name]["sum"]/classes_dict[class_name]["cpt"],2)
        i = 0
        while i<len(l):
            l[i]=classes_dict[out[i]]["moy"]
            i+=1
    return out

def div_to_k_width(df, k=None, classes=None,sur= 0):
    """
    The function `div_to_k_width` divides a given dataframe into k classes based on a specified column,
    and assigns class labels to each row.
    
    :param df: The input dataframe, where each row represents a data point and each column represents a
    feature
    :param k: The number of classes or groups to divide the data into. If not provided, it will be
    computed using the `compute_k` function
    :param classes: The `classes` parameter is a list of strings that represents the class labels for
    the data. Each element in the list corresponds to a class label for a specific range of values. The
    length of the `classes` list should be equal to the value of `k`, which represents the number of
    classes
    :param sur: The parameter "sur" is used to specify the index of the column in the dataframe that you
    want to use for sorting the data, defaults to 0 (optional)
    :return: The function `div_to_k_width` returns a modified version of the input dataframe `df` where
    the values in the specified column `sur` have been divided into `k` classes. The modified dataframe
    is returned as a list.
    """
    if k is None:
        Temperature = [i[0] for i in df[1:]]
        k = compute_k(Temperature)
    if classes is None or len(classes) != k:
        classes = ["class_" + str(i + 1) for i in range(k)]
    
    # Sort the input list
    sorted_l = sorted(df[1:],key = lambda x : x[sur])
    n = len(sorted_l)
    class_size = n // k
    remainder = n % k
    i = 0
    out = [df[0]]
    

    for class_n in range(k):
        class_start = i
        class_end = i + class_size + (1 if class_n < remainder else 0)
        class_values = sorted_l[class_start:class_end]

        for val in class_values:
            val[sur] = classes[class_n]
            out.append(val)
        i = class_end

    return out

def C(data,k=2,cmin = 2):
    """
    The function C takes in a dataset and finds frequent itemsets using the Apriori algorithm.
    
    :param data: The "data" parameter is a dictionary where the keys represent transactions and the
    values represent the items in each transaction. Each transaction is represented as a list of items
    :param k: The parameter "k" represents the number of frequent itemsets to be generated. In this
    case, it is set to 2, which means that the algorithm will generate frequent itemsets of size 2,
    defaults to 2 (optional)
    :param cmin: The parameter `cmin` represents the minimum support count for an item to be considered
    frequent. It is used to filter out infrequent items from the dataset, defaults to 2 (optional)
    :return: The function `C` returns the result of the `cn_func` function.
    """
    c1 = dict()
    for l in data.values():
        for i in l:
            if c1.get(i):
                c1[i]+=1
            else:
                c1[i] = 1
    print("c1 :",c1)
    print("nombre des items :" , len(c1))
    l1 = l_func(c1,cmin)
    return cn_func(l1,l1,data,k-1,2,[l1],cmin)
def l_func(c,cmin=2):
    """
    The function l_func takes a dictionary c and a minimum count cmin as input, and returns a new
    dictionary l1 that contains only the key-value pairs from c where the value is greater than or equal
    to cmin.
    
    :param c: A dictionary where the keys are items and the values are their corresponding counts
    :param cmin: The parameter `cmin` is the minimum value that a key-value pair in the dictionary `c`
    must have in order to be included in the resulting dictionary `l1`, defaults to 2 (optional)
    :return: a dictionary containing the items from the input dictionary `c` that have a value greater
    than or equal to `cmin`.
    """
    l1 = dict()
    for i,j in c.items():
        if j>=cmin:
            l1[i]=j
    return l1
def get_new_item_list(l1,l2,n):
    """
    The function `get_new_item_list` takes two lists `l1` and `l2`, and an integer `n`, and returns a
    new list that contains all possible combinations of elements from `l1` and `l2` where each
    combination has `n` elements and no element from `l2` is already present in the combination.
    
    :param l1: The parameter l1 is a list of strings
    :param l2: The parameter `l2` is a list of items that can be added to the items in `l1`
    :param n: The parameter `n` represents the number of items to be combined from the two lists `l1`
    and `l2`
    :return: a list of new items created by combining elements from two input lists, l1 and l2. The
    number of new items created is determined by the value of n.
    """
    if n>2:
        new_list= []
        for i in l1 :
            for j in l2:
                if j not in i.split(","):
                    new_list.append(",".join([i,j]))
        return get_new_item_list(new_list,l2,n-1)
    else:
        new_list= []
        for i in l1 :
            for j in l2:
                if j not in i.split(","):
                    new_list.append(",".join([i,j]))
        return new_list

def cn_func(l,l1,data,k,n,l_list,cmin):
    """
    The function `cn_func` takes in a set of data and performs a specific calculation on it, returning
    the result along with a list of intermediate results.
    
    :param l: The parameter `l` is a dictionary that represents the frequent itemsets of length `k-1`.
    Each key in the dictionary is a frequent itemset, and the corresponding value is the support count
    of that itemset
    :param l1: The parameter `l1` is a dictionary that contains the frequent itemsets of length `k-1`
    :param data: The `data` parameter is a dictionary that contains the data for which we want to find
    frequent itemsets. Each key in the dictionary represents a transaction, and the corresponding value
    is a list of items in that transaction
    :param k: The parameter `k` represents the number of iterations or levels of the algorithm. It
    determines how many times the algorithm will recursively call itself
    :param n: The parameter `n` represents the length of the itemsets that we are interested in finding.
    It is initially set to 1 and is incremented by 1 in each recursive call of the `cn_func` function
    :param l_list: The `l_list` parameter is a list that stores the frequent itemsets generated by the
    algorithm
    :param cmin: The parameter `cmin` is not defined in the given code snippet. It seems to be used as
    an argument in the `l_func` function, but without knowing the implementation of `l_func`, it is
    difficult to determine its purpose. Please provide more information or the implementation of `l_func
    :return: two values: `l` and `l_list`.
    """
    print(k)
    if k > 0:
        cn = dict()
        lv = list(l1.keys())
        list_of_newkeys= get_new_item_list(lv,lv,n)

        for i in  list_of_newkeys:
            li = i.split(",")
            
            for l in data.values():
                f = True
                for e in li: 
                    if not(e in l):
                        f = False
                if f:
                    if cn.get(i):
                        cn[i]+=1
                    else:
                        cn[i] = 1
        ln =l_func(cn,cmin)
        l_list.append(ln)
        return cn_func(ln,l1,data,k-1,n+1,l_list,cmin)
    else :
        return l, l_list

def get_cmbinations(s):
    """
    The function `get_combinations` takes a string as input and returns a list of all possible
    combinations of substrings from the input string.
    
    :param s: The parameter `s` is a string that contains comma-separated values
    :return: The function `get_combinations` returns a list of combinations of elements from the input
    string `s`. Each combination is represented as a list of two sublists: the first sublist contains
    the elements selected for the combination, and the second sublist contains the remaining elements.
    """
    out = []
    l = s.split(",")
    for pos in range(len(l)):
        for i in range(pos+1,len(l)):
            a = l[pos:i]
            try:
                b1=l[:pos]
            except:
                b1=[]
            try:
                b2=l[i:]
            except:
                b2=[]
            b = b1+b2
            out.append([a,b])
    return out

def RA(lks):
    """
    The function RA takes a list of dictionaries as input and returns a list of all possible
    combinations of the keys in the dictionaries.
    
    :param lks: The parameter `lks` is a list of dictionaries. Each dictionary represents a set of
    key-value pairs, where the keys are strings and the values can be any data type
    :return: a list of all possible combinations of the keys in the dictionaries within the input list.
    """
    all_RA = []
    for lk in lks:
        for s in lk.keys():
            all_RA.extend(get_cmbinations(s))
    return all_RA

def compute_the_belive(RA,lks):
    """
    The function `compute_the_belive` calculates the belief of a given set of attributes in a given set
    of transactions.
    
    :param RA: RA is a list containing two sublists. The first sublist represents the items that need to
    be present in a transaction for it to be considered for calculation. The second sublist represents
    the items that need to be present in addition to the items in the first sublist for the transaction
    to be considered for calculation
    :param lks: lks is a dictionary where the keys are strings representing transactions and the values
    are dictionaries representing the items in each transaction and their corresponding support counts
    :return: the ratio of the number of transactions that satisfy both conditions in RA[0] and RA[1] to
    the number of transactions that satisfy only the condition in RA[0].
    """
    lk = lks[len(RA[0])+len(RA[1])-1]
    #check A
    cpt_a = 0
    cpt_b = 0
    for i in lk.keys():
        trans = i.split(",")
        flag = True
        for item in RA[0]:
            if not item in trans:
                flag = False
        if flag : 
            cpt_a += 1
            for item in RA[1]:
                if not item in trans:
                    flag = False
            if flag:# still true
                cpt_b += 1
    return cpt_b/cpt_a