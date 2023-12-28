import streamlit as st
from utils import *
from models import *


# The above code is creating a dictionary called `f_dist_dict` with four key-value pairs. The keys are
# strings representing different distance measures, such as "Ecludian distance", "Cos distance",
# "Manhatan distance", and "Minko distance". The values are variables representing the corresponding
# distance calculations, such as `dest_eclude`, `dest_cos`, `dest_manhat`, and `dest_minko`.
f_dist_dict ={
                "Ecludian distance":dest_eclude,
                "Cos distance":dest_cos,
                "Manhatan distance":dest_manhat,
                "Minko distance":dest_minko
            }

# The above code is defining a dictionary called `tree_mode_dict` with two key-value pairs. The keys
# are "repeat attributes" and "normal", and the values are 1 and 0 respectively. This dictionary is
# used to store different modes for a tree.
tree_mode_dict = {
    "repeat attributes":1,
    "normal":0
}

# The above code is defining a dictionary called `columns_dict` that maps column names to their
# corresponding indices. This dictionary is used to easily access the index of a column when working
# with data in a tabular format.
columns_dict = {
    "N":0,
    "P":1,
    "K":2,
    "pH":3,
    "EC":4,
    "OC":5,
    "S":6,
    "Zn":7,
    "Fe":8,
    "Cu":9,
    "Mn":10,
    "B":11,
    "OM":12,
    "Fertility":13
}

# The above code is creating a list called `pred_to_label` which contains three elements: "faiblement
# (low)", "moyennement (medium)", and "fortement (high)". This list is likely used to map predicted
# values to corresponding labels in a machine learning or data analysis task.
pred_to_label = ["faiblement (low)", "moyennement (medium)", "fortement (high)" ]

# The above code is checking if the key 'stage' is not present in the session state dictionary. If it
# is not present, it initializes it with a value of 0.

if 'stage' not in st.session_state:
    st.session_state.stage = 0

# The above code is checking if the key 'stage_c' is not present in the st.session_state dictionary.
# If it is not present, it initializes it with a value of 0.
if 'stage_c' not in st.session_state:
    st.session_state.stage_c = 0

def set_state_c(i):
    """
    The function `set_state_c` sets the value of the `stage_c` variable in the `st.session_state`
    object.
    
    :param i: The parameter "i" is the value that you want to set for the session state variable
    "stage_c"
    """
    st.session_state.stage_c = i

st.title("DATAMINING PROJECT")

data = st.selectbox("Dataset",["Dataset1","Dataset3"])
if data=="Dataset1":
    st.button("refrash")
    df = load_data_par_row("Dataset1.csv",columns = ["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM","Fertility"],
                       col_type = [float,float,float,float,float,float,float,float,float,float,float,float,float,int])
    

    if st.toggle("remove missing values",value=True):
        fixWithMean(df)
    
    if st.toggle("fix abberate values",value=True):
        fixAbb(df)
    if st.toggle("remove horizontal redondancy",value=True):
        df = remove_hozizontale(df)
    
    norm = st.selectbox("Normalization",["None","MinMax","Z-Score","MinMax + S-Score"])
    if norm == "MinMax":
        df ,min_norm,max_norm,mean_norm,std_norm = normHist(df,Z=False,plot=False,do_not=[13])
    elif norm == "Z-Score":
        df ,min_norm,max_norm,mean_norm,std_norm = normHist(df,MM=False,plot=False,do_not=[13])
    elif "MinMax + S-Score":
        df,min_norm,max_norm,mean_norm,std_norm = normHist(df,plot=False,do_not=[13])

    seed =  st.number_input("seed",value=32)
    test_per = st.slider("test size %",0,100,value=20)/100
    df_train ,df_test,x_train,x_test,y_train,y_test = train_test_split(df,seed=seed,test_per=test_per)
    st.subheader("Model")
    task = st.radio("Task",["Classification","Clusturing"],on_change=set_state_c, args=[0])
    if task=="Classification":
        model_type = st.radio("model",["KNN","Decision Tree","Random Forest"],on_change=set_state_c, args=[0])
        if model_type=="KNN":
            k = st.number_input("k",1)
            f_dist = st.radio("distance function",["Ecludian distance","Cos distance","Manhatan distance","Minko distance"])
            if st.session_state.stage_c >= 0:
                st.button("train",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Training...'):
                    st.session_state.model = KNN(df_train,k=k,f_dist=f_dist_dict[f_dist])
                with st.spinner('Testing...'):
                    y_pred = st.session_state.model.predict_all(x_test)
                    eval = Evaluator(y_test,y_pred,[0,1,2])
                    st.session_state.to_show = "\n".join(eval.report())
                    st.session_state.stage_c = 2
            if st.session_state.stage_c == 2:
                st.text(st.session_state.to_show)
                st.subheader("Input")
                n = st.number_input('Nitrogen (N)', min_value=0.0)
                p = st.number_input('Phosphorus (P)', min_value=0.0)
                k = st.number_input('Potassium (K)', min_value=0.0)
                ph = st.number_input('pH', min_value=0.0, max_value=14.0)
                ec = st.number_input('Electrical Conductivity (EC)', min_value=0.0)
                oc = st.number_input('Organic Carbon (OC)', min_value=0.0)
                s = st.number_input('Sulfur (S)', min_value=0.0)
                zn = st.number_input('Zinc (Zn)', min_value=0.0)
                fe = st.number_input('Iron (Fe)', min_value=0.0)
                cu = st.number_input('Copper (Cu)', min_value=0.0)
                mn = st.number_input('Manganese (Mn)', min_value=0.0)
                b = st.number_input('Boron (B)', min_value=0.0)
                om = st.number_input('Organic Matter (OM)', min_value=0.0)
                if st.button("predict"):
                    input_data = [n,p,k,ph,ec,oc,s,zn,fe,cu,mn,b,om]
                    norm_data = []
                    for i,v in enumerate(input_data):
                        if len(min_norm)>0:
                            v= (v-min_norm[i])/(max_norm[i]-min_norm[i])
                        if len(mean_norm)>0:
                            v= (v-mean_norm[i])/(std_norm[i])
                        norm_data.append(v)
                    st.success(f"le sol est {pred_to_label[st.session_state.model.predict(norm_data)] } fertile.")


        elif model_type=="Decision Tree":
            min_leaf_size = st.number_input("min leaf size",1)
            max_depth = st.number_input("max depth",1,value=10)
            tree_mode = st.radio("mode",["repeat attributes","normal"])
            if st.session_state.stage_c >= 0:
                st.button("train",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Training...'):
                    st.session_state.model = DTree(df_train,'Fertility',min_leaf_size=min_leaf_size,max_depth=max_depth,mode=tree_mode_dict[tree_mode])
                with st.spinner('Testing...'):
                    y_pred = st.session_state.model.predict_all(x_test)
                    eval = Evaluator(y_test,y_pred,[0,1,2])
                    st.session_state.to_show = "\n".join(eval.report())
                    st.session_state.stage_c = 2
            if st.session_state.stage_c == 2:
                st.text(st.session_state.to_show)
                st.subheader("Input")
                n = st.number_input('Nitrogen (N)', min_value=0.0)
                p = st.number_input('Phosphorus (P)', min_value=0.0)
                k = st.number_input('Potassium (K)', min_value=0.0)
                ph = st.number_input('pH', min_value=0.0, max_value=14.0)
                ec = st.number_input('Electrical Conductivity (EC)', min_value=0.0)
                oc = st.number_input('Organic Carbon (OC)', min_value=0.0)
                s = st.number_input('Sulfur (S)', min_value=0.0)
                zn = st.number_input('Zinc (Zn)', min_value=0.0)
                fe = st.number_input('Iron (Fe)', min_value=0.0)
                cu = st.number_input('Copper (Cu)', min_value=0.0)
                mn = st.number_input('Manganese (Mn)', min_value=0.0)
                b = st.number_input('Boron (B)', min_value=0.0)
                om = st.number_input('Organic Matter (OM)', min_value=0.0)
                if st.button("predict"):
                    input_data = [n,p,k,ph,ec,oc,s,zn,fe,cu,mn,b,om]
                    norm_data = []
                    for i,v in enumerate(input_data):
                        if len(min_norm)>0:
                            v= (v-min_norm[i])/(max_norm[i]-min_norm[i])
                        if len(mean_norm)>0:
                            v= (v-mean_norm[i])/(std_norm[i])
                        norm_data.append(v)
                    st.success(f"le sol est {pred_to_label[st.session_state.model.predict(norm_data)] } fertile.")


        elif model_type=="Random Forest":
            n_estimators = st.number_input("n_estimators",1)
            start_with_att = st.number_input("start with how many of attributes",1)
            min_leaf_size = st.number_input("min leaf size",1)
            max_depth = st.number_input("max depth",1,value=10)
            tree_mode = st.radio("mode",["repeat attributes","normal"])
            if st.session_state.stage_c >= 0:
                st.button("train",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Training...'):
                    st.session_state.model = RandomForest(df_train,'Fertility',start_with_att=start_with_att,n_estimators=n_estimators,min_leaf_size=min_leaf_size,max_depth=max_depth,mode=tree_mode_dict[tree_mode])
                with st.spinner('Testing...'):
                    y_pred = st.session_state.model.predict_all(x_test)
                    eval = Evaluator(y_test,y_pred,[0,1,2])
                    st.session_state.to_show = "\n".join(eval.report())
                    st.session_state.stage_c = 2
            if st.session_state.stage_c == 2:
                st.text(st.session_state.to_show)
                st.subheader("Input")
                n = st.number_input('Nitrogen (N)', min_value=0.0)
                p = st.number_input('Phosphorus (P)', min_value=0.0)
                k = st.number_input('Potassium (K)', min_value=0.0)
                ph = st.number_input('pH', min_value=0.0, max_value=14.0)
                ec = st.number_input('Electrical Conductivity (EC)', min_value=0.0)
                oc = st.number_input('Organic Carbon (OC)', min_value=0.0)
                s = st.number_input('Sulfur (S)', min_value=0.0)
                zn = st.number_input('Zinc (Zn)', min_value=0.0)
                fe = st.number_input('Iron (Fe)', min_value=0.0)
                cu = st.number_input('Copper (Cu)', min_value=0.0)
                mn = st.number_input('Manganese (Mn)', min_value=0.0)
                b = st.number_input('Boron (B)', min_value=0.0)
                om = st.number_input('Organic Matter (OM)', min_value=0.0)
                if st.button("predict"):
                    input_data = [n,p,k,ph,ec,oc,s,zn,fe,cu,mn,b,om]
                    norm_data = []
                    for i,v in enumerate(input_data):
                        if len(min_norm)>0:
                            v= (v-min_norm[i])/(max_norm[i]-min_norm[i])
                        if len(mean_norm)>0:
                            v= (v-mean_norm[i])/(std_norm[i])
                        norm_data.append(v)
                    st.success(f"le sol est {pred_to_label[st.session_state.model.predict(norm_data)] } fertile.")


    else:
        model_type = st.radio("model",["K-Means","DBSCAN","Cheat Mode"],on_change=set_state_c, args=[0])
        if model_type=="K-Means":
            k = st.number_input("k",1)
            max_iter = st.number_input("max iter",1)
            epsilon = st.number_input("epsilon",0.0)
            f_dist = st.radio("distance function",["Ecludian distance","Cos distance","Manhatan distance","Minko distance"])
            trained = False
            if st.session_state.stage_c >= 0:
                st.button("train",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Training...'): 
                    st.session_state.model = k_means(x_train,max_iter=max_iter,epsilon=epsilon,k=k,f_dist=f_dist_dict[f_dist])
                with st.spinner('Testing...'): 
                    st.text(f"silhouette coefficient {silhouette_coefficient(st.session_state.model.clusters):.2f}")
                st.session_state.stage_c = 2
            if st.session_state.stage_c == 2:
                x1 = st.selectbox("X1",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 4)
                x2 = st.selectbox("X2",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 5)
                with st.spinner('Ploting...'): 
                    st.pyplot(plot_clusters(st.session_state.model.clusters,columns_dict[x1],columns_dict[x2],columns1_name=x1,columns2_name=x2))
                
                
        elif model_type=="DBSCAN":
            min_pts = st.number_input("min pts",0,value=10)
            epsilon = st.number_input("epsilon",0.0,value=1.4)
            f_dist = st.radio("distance function",["Ecludian distance","Cos distance","Manhatan distance","Minko distance"])
            trained = False
            if st.session_state.stage_c >= 0:
                st.button("train",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Training...'): 
                    st.session_state.model = DBSCAN(x_train,min_pts=min_pts,epsilon=epsilon,f_dist=f_dist_dict[f_dist])
                with st.spinner('Testing...'): 
                    st.text(f"silhouette coefficient {silhouette_coefficient(st.session_state.model.clusters):.2f}")
                    st.session_state.stage_c = 2
            if st.session_state.stage_c == 2:
                x1 = st.selectbox("X1",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 4)
                x2 = st.selectbox("X2",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 5)
                with st.spinner('Ploting...'): 
                    st.pyplot(plot_clusters(st.session_state.model.clusters,columns_dict[x1],columns_dict[x2],columns1_name=x1,columns2_name=x2))
                
                
        elif model_type=="Cheat Mode": 
            the_pertect_clusters = [[i[:-1] for i in df_train if isinstance(i[-1],str) or (i[-1]==j)] for j in [0,1,2]]
            if st.session_state.stage_c >= 0:
                st.button("test",on_click=set_state_c, args=[1])
            if st.session_state.stage_c == 1:
                with st.spinner('Testing...'):
                    st.text(f"silhouette coefficient {silhouette_coefficient(the_pertect_clusters):.2f}")
                    st.session_state.stage_c =2
            if st.session_state.stage_c ==2:
                x1 = st.selectbox("X1",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 4)
                x2 = st.selectbox("X2",["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B","OM"],index= 5)
                with st.spinner('Ploting...'):
                    st.pyplot(plot_clusters(the_pertect_clusters,columns_dict[x1],columns_dict[x2],columns1_name=x1,columns2_name=x2))
            
else:
    def set_state(i):
        st.session_state.stage = i
    st.button("refrash")
    df3 = load_data_par_row("Dataset3_fixed.csv",columns = ["Temperature","Humidity","Rainfall","Soil","Crop","Fertilizer"],
                       col_type = [float,float,float,str,str,str],rm_end=True)
    div_to_k_width(df3,classes=["Freezing","Cold","Chilly","Cool","Warm","Hot","Scorching","Sweltering","Searing"],sur=0)
    div_to_k_width(df3,classes=["humidity_" + str(i + 1) for i in range(9)],sur=1)
    div_to_k_width(df3,classes=["Rainfall_" + str(i + 1) for i in range(9)],sur=2)

    k = st.number_input("k",1,len(df3[0]),value=6)
    supp_min = st.slider("support minimum",0,100,value=15)/100
    conf_min = st.slider("confiance minimum",0,100,value=50)
    ready = False
    if st.session_state.stage >= 0:
        st.button('Run', on_click=set_state, args=[1])
    if st.session_state.stage==1:
        with st.spinner('Running...'):
            lk,lks=C({i:j for i,j in enumerate(df3[1:])},k=6,cmin = int(len(df3)*supp_min))
            tout_les_RA = RA(lks)
            st.session_state.confRA = []
            st.session_state.to_show = []
            for i in tout_les_RA:
                conf = round(compute_the_belive(i,lks)*100,2)
                if conf>=conf_min:
                    st.session_state.confRA.append([i,conf])
                    st.session_state.to_show.append(f" {i[0]} --> {i[1]} | confiance = {conf}%")
        st.session_state.stage=2
    if st.session_state.stage>=2:
        st.text("\n".join(st.session_state.to_show))
        #input 
        st.subheader("Input")
        temperature = st.selectbox("Temperature",["Freezing","Cold","Chilly","Cool","Warm","Hot","Scorching","Sweltering","Searing"])
        humidity = st.selectbox("Humidity",["humidity_" + str(i + 1) for i in range(9)])
        rainfall = st.selectbox("Rainfall",["Rainfall_" + str(i + 1) for i in range(9)])
        soil = st.selectbox('Soil',['Clayey', 'laterite', 'silty clay', 'sandy', 'coastal','clay loam', 'alluvial'])
        crop = st.selectbox('Crop',['rice', 'Coconut'])
        fertilizer = st.selectbox('Fertilizer',['DAP', 'Good NPK', 'MOP', 'Urea'])
        
        if st.button("predict",):
            input_data =  [temperature,humidity,rainfall,soil,crop,fertilizer]
            out = []
            print(input_data)
            for i in st.session_state.confRA:
                print(i)
                yes = True
                for prem in i[0][0]:
                    if prem not in input_data:
                        yes = False
                if yes:
                    out.extend(i[0][1])
            print(out)
            for i in out:
                st.success(i)