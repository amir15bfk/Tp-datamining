import math
import random
from  utils import *


# The above class is a K-nearest neighbors (KNN) classifier that can be used to predict the class
# labels of new data points based on the distances to the nearest neighbors in a given dataset.
class KNN:
    def __init__(self,df,k=3,f_dist = dest_eclude):
        """
        The function initializes an object with a dataframe, a value for k, and a distance function.
        
        :param df: The parameter `df` is a variable that represents a dataframe. It is used to store and
        manipulate data in a tabular format
        :param k: The parameter "k" represents the number of nearest neighbors to consider when
        performing a k-nearest neighbors algorithm. In this case, it is set to 3, meaning that the
        algorithm will consider the 3 closest neighbors to a given data point, defaults to 3 (optional)
        :param f_dist: The parameter `f_dist` is a function that calculates the distance between two
        data points. It is set to `dest_eclude` by default
        """
        self.df = df
        self.k = k 
        self.f_dist = f_dist
    def predict(self,v):
        return find_dom_in_k([i[1] for i in self.ord_dest_vect(v)[:self.k]])
    def predict_all(self,X):
        preds = []
        for v in X[1:]:
            preds.append(self.predict(v))
        return preds
    def ord_dest_vect(self,v):
        ord_vect =[]
        for target in self.df[1:]:
            d = self.f_dist(v,target[:-1])
            ord_vect.append([d,target[-1]])
        ord_vect.sort()
        return ord_vect

# The `DTree` class is a decision tree implementation in Python that can be used for classification
# tasks.
class DTree:
    
    def __init__(self,dataset,taget_attribute,attributes=None,spacing=1,min_leaf_size=4,max_depth = 10,mode=0):
        self.dataset = dataset
        if isinstance(taget_attribute,str):
            self.taget_attribute = [(idx,name) for idx,name in enumerate(dataset[0]) if name==taget_attribute][0]
        else:
            self.taget_attribute = taget_attribute
        if attributes == None:
            self.attributes = [(idx,name) for idx,name in enumerate(dataset[0]) if name!=self.taget_attribute[1]]
        else:
            self.attributes = attributes
        self.spacing = spacing
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.mode = mode
        # print("len dataset:",len(dataset))
        # print(self.attributes)
        self.make_the_tree()
    
    def entropy(self):
        self.N = len(self.dataset)-1
        self.tcount_dict = dict()
        for i in self.dataset[1:]:
            if self.tcount_dict.get(i[self.taget_attribute[0]]):
                self.tcount_dict[i[self.taget_attribute[0]]]+=1
            else:
                self.tcount_dict[i[self.taget_attribute[0]]]=1
        self.entropyD = 0
        for i in self.tcount_dict.values():
            self.entropyD -= ((i/self.N) * math.log2(i/self.N))
    
    def entropyA(self,att):
        self.N = len(self.dataset)-1
        sorted_dataset = sorted(self.dataset[1:],key=lambda x:x[att])
        best_entropyA = 9999
        best_thresh = None
        for j in range(0,1):#len(sorted_dataset)-self.spacing,self.spacing):
            thresh = (sorted_dataset[j][att]+sorted_dataset[-1][att])/2
            # print(thresh)
            att_top_dict = dict()
            n_top = 0 
            att_under_dict = dict()
            n_under = 0
            
            for i in sorted_dataset:
                if i[att]>=thresh:
                    n_top+=1
                    if att_top_dict.get(i[self.taget_attribute[0]]):
                        att_top_dict[i[self.taget_attribute[0]]]+=1
                    else:
                        att_top_dict[i[self.taget_attribute[0]]]=1
                else:
                    n_under+=1
                    if att_under_dict.get(i[self.taget_attribute[0]]):
                        att_under_dict[i[self.taget_attribute[0]]]+=1
                    else:
                        att_under_dict[i[self.taget_attribute[0]]]=1
            entropyAtt_top = 0
            for i in att_top_dict.values():
                entropyAtt_top -= ((i/n_top) * math.log2(i/n_top))
            entropyAtt_under = 0
            for i in att_under_dict.values():
                
                entropyAtt_under -= ((i/n_under) * math.log2(i/n_under))
            entropyA  = self.entropyD - (n_top/self.N)*entropyAtt_top - (n_under/self.N)*entropyAtt_under 
            if entropyA<best_entropyA:
                best_entropyA = entropyA
                best_thresh = thresh
        return best_entropyA , best_thresh

    def make_leafs(self):
        #print(self.attribute_check)
        if self.mode ==0:
            attributes_rest = [(idx,name) for idx,name in self.attributes if idx!=self.attribute_check]
        elif self.mode==1:
            attributes_rest = self.attributes

        dataset_left = [self.dataset[0]]
        dataset_right = [self.dataset[0]]
        for i in self.dataset[1:]:
            if i[self.attribute_check]>=self.thresh:
                dataset_left.append(i)
            else:
                dataset_right.append(i)
        # print(self.dataset)
        # print(dataset_left)
        # print(dataset_right)
        if (len(dataset_left)<2) or (len(dataset_right)<2):
            self.state = 0
            self.val = max(list(self.tcount_dict.items()),key=lambda x:x[1])[0]
            return 0 
        self.left = DTree(dataset_left,self.taget_attribute,attributes_rest,spacing=self.spacing,min_leaf_size=self.min_leaf_size,max_depth=self.max_depth-1)

        self.right = DTree(dataset_right,self.taget_attribute,attributes_rest,spacing=self.spacing,min_leaf_size=self.min_leaf_size,max_depth=self.max_depth-1)
    def make_the_tree(self):
        self.entropy()
        if self.entropyD<0:
            print("error")
        if (len(self.dataset)<self.min_leaf_size) or (self.entropyD<=0) or (len(self.attributes)==0) or (self.max_depth<=1) :
            self.state = 0
            # print(self.entropyD,self.N,self.tcount_dict,self.dataset)
            self.val = max(list(self.tcount_dict.items()),key=lambda x:x[1])[0]
            return 0
        entropys = []
        for idx_att,_ in self.attributes:
            best_entropyA , best_thresh = self.entropyA(idx_att)
            entropys.append((best_entropyA , best_thresh,idx_att))
        entropys.sort(key=lambda x:x[0])
        
        _ ,self.thresh,self.attribute_check = entropys[0]
        # print(entropys,self.attribute_check)
        self.state = 1
        self.make_leafs()
    
    def predict(self,x):
        if self.state==0:
            return self.val
        else:
            if x[self.attribute_check]>=self.thresh:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
    
    def predict_all(self,X):
        predicts = []
        for i in X[1:]:
            predicts.append(self.predict(i))
        return predicts
    
    def print_tree(self):
        if self.state==0:
            print(f" end : {self.val} ")
        else:
            print(f" check this to {self.attribute_check}>={self.thresh}:")
            self.left.print_tree()
            self.right.print_tree()

    
# The RandomForest class is a Python implementation of the random forest algorithm for classification
# and regression tasks.
class RandomForest:
    def __init__(self,df,taget_attribute,start_with_att=5,mode = 1,max_depth = 10,n_estimators = 20,spacing=1,min_leaf_size=1):
        self.df = df
        self.start_with_att = start_with_att
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.spacing = spacing
        self.min_leaf_size = min_leaf_size
        self.mode = mode
        self.taget_attribute = [(idx,name) for idx,name in enumerate(self.df[0]) if name==taget_attribute][0]
        self.attributes = [(idx,name) for idx,name in enumerate(self.df[0]) if name!=self.taget_attribute[1]]
        self.create_the_trees()
    
    def create_the_trees(self):
        self.trees = []
        for i in range(self.n_estimators):
            booststapped_dataset = self.get_booststapped()
            booststapped_attributes = random.sample(self.attributes,self.start_with_att)
            self.trees.append(DTree(booststapped_dataset,self.taget_attribute,booststapped_attributes,spacing=self.spacing,min_leaf_size=self.min_leaf_size,max_depth=self.max_depth,mode=self.mode))
    
    def get_booststapped(self):
        booststapped_dataset = [self.df[0]]
        booststapped_dataset+=random.choices(self.df[1:],k=len(self.df)-1)
        return booststapped_dataset

    def predict(self,x):
        return find_dom_in_k([tree.predict(x) for tree in self.trees])

    def predict_all(self,X):
        predicts = []
        for i in X[1:]:
            predicts.append(self.predict(i))
        return predicts


# The above class implements the k-means clustering algorithm in Python.
class k_means:
    def __init__(self,df,k=3,f_dist=dest_eclude,epsilon = 0.01,max_iter=10,mode=0):
        self.df = df 
        self.k = k
        self.f_dist = f_dist
        self.centers = random.sample(self.df[1:],self.k)
        print(self.centers)
        for i in range(max_iter):
            self.build_clusters()
            delta_center = self.update_centers()
            if delta_center < epsilon:
                break
    
    def build_clusters(self):
        self.clusters = []
        for i in range(len(self.centers)):
            self.clusters.append([self.df[0]])
        for v in self.df[1:]:
            dv =[]
            for i in range(len(self.centers)):
                dv.append([self.f_dist(v,self.centers[i]),i])
            self.clusters[min(dv,key = lambda x:x[0])[1]].append(v)

    def update_centers(self):
        delta_center = 0
        for i,cluster in enumerate(self.clusters):
            print(i," : ", len(cluster))
            c = self.center(cluster)
            delta_center += self.f_dist(self.centers[i],c)
            self.centers[i] = c
        return delta_center
    
    def center(self,cluster):
        c = []
        for col in range(len(cluster[0])):
            l = mean([row[col] for row in cluster[1:] if row[col]!=None])
            c.append(l)
        return c

# The DBSCAN class is an implementation of the Density-Based Spatial Clustering of Applications with
# Noise algorithm in Python.
class DBSCAN:
    def __init__(self,df,epsilon=1,min_pts=2,f_dist = dest_eclude):
        self.df = df
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.f_dist = f_dist
        self.P = dict()
        # init p  
        for p,_ in enumerate(self.df[1:]):
            self.P[p]={"status":0,"cluster":0}
        self.bruit = []
        self.c = 0
        for p,_ in enumerate(self.df[1:]):
            if self.P[p]["status"] != 1 :
                self.P[p]["status"] = 1
                pts_voisions = self.epsiloneVoisinage(p)
                if len(pts_voisions)<self.min_pts:
                    self.P[p]["cluster"] = None
                else:
                    self.c+=1
                    self.etendreCluster(p,pts_voisions)
        self.clusters = []
        for i in range(self.c):
            self.clusters.append([self.df[0]])
        for i,j in self.P.items():
            if j["cluster"]!=None:
                self.clusters[j["cluster"]-1].append(df[i+1])


    def epsiloneVoisinage(self,p):
        out =[]
        for pv,_ in enumerate(self.df[1:]):
            if p!=pv:
                try:
                    if self.f_dist(self.df[p+1],self.df[pv+1])<=self.epsilon:
                        out.append(pv)
                except:
                    print(self.df[p],self.df[pv])
        return out

    def etendreCluster(self,p,pts_voisions):
        self.P[p]["cluster"]=self.c
        for p_hat in pts_voisions:
            if self.P[p_hat]["status"] != 1:
                self.P[p_hat]["status"] = 1
                pts_voisions_hat = self.epsiloneVoisinage(p_hat)
                if len(pts_voisions_hat)>=self.min_pts:
                    pts_voisions = list(set(pts_voisions+pts_voisions_hat))
            if self.P[p_hat]["status"] == 1:
                self.P[p_hat]["cluster"] = self.c
