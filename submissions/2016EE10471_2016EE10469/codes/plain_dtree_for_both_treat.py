import pandas as pd

df  = pd.read_csv('data.csv')
df2 = pd.read_csv('newdata2.csv')
df3 = pd.read_csv('mar_data.csv')


n = len(df2['girls2'])
n_list = list(df2['morekids'])


more = []
nomore = []
for i in range(n):
    if n_list[i] == True:
        more.append(i)
    else:
        nomore.append(i)

df_more = df2.iloc[more,:].copy()
df_nomore = df2.iloc[nomore,:].copy()



n3 = len(df3['girls2'])
n3_list = list(df3['morekids'])


more = []
nomore = []
for i in range(n3):
    if n3_list[i] == True:
        more.append(i)
    else:
        nomore.append(i)

df_mar_more = df3.iloc[more,:].copy()
df_mar_nomore = df3.iloc[nomore,:].copy()


df_nomore['incomem']=(df_nomore['incomem']).fillna(0)


from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import numpy as np
from sklearn.tree import export_graphviz 
import collections
import pydotplus


from sklearn.tree import DecisionTreeRegressor   
regressor = DecisionTreeRegressor(random_state = 0, max_depth=3)  
y1 = ['workedm','weeksm1','hourswm','incomem','famincl']
y2 = ['workedm','weeksm1','hourswm','incomem','famincl', 'workedd','weeksd1','hourswd','incomed','nonmomil']
features = np.asarray(['agem1','agefstm','educm','blackm','whitem','hispm','othracem'])



for m in y2:
    regressor.fit(np.asarray(df_mar_more[['agem1','agefstm','educm','blackm','whitem','hispm','othracem']]), np.asarray(df_mar_more[m]))
    dot_data = tree.export_graphviz(regressor,
                                    feature_names=features,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    print(m)
    s = m+".png"
    
    graph.write_png(s)


# In[ ]:


for m in y1:
    regressor.fit(np.asarray(df_more[['agem1','agefstm','educm','blackm','whitem','hispm','othracem']]), np.asarray(df_more[m]))
    dot_data = tree.export_graphviz(regressor,
                                    feature_names=features,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()    
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    print(m)
    s = m+".png"
    
    graph.write_png(s)

