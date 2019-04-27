import pandas as pd

df  = pd.read_csv('data.csv')
df2 = pd.read_csv('newdata2.csv')
df3 = pd.read_csv('mar_data.csv')


p = df2['morekids'].mean()



n = len(df2['girls2'])
n_list = list(df2['morekids'])



df2['incomem']=(df2['incomem']).fillna(0)


al = ['morekids', 'workedm','weeksm1','hourswm','incomem','famincl']
for i in al:
    df2[i] = (df2[i]).astype(int)


al_mar = ['morekids','workedm','weeksm1','hourswm','incomem','famincl', 'workedd','weeksd1','hourswd','incomed','nonmomil']
for i in al:
    df3[i] = (df3[i]).astype(int)


al2 = ['workedm','weeksm1','hourswm','incomem','famincl']
for i in al2:
    ln = df2[i]
    listlist=[]
    for j in range(n):
        if n_list[j] ==1:
            listlist.append((1/p)*ln[j])
        else:
            listlist.append((ln[j]/(p-1)))
    df2[i+"pp"] = listlist
    print(i)


n3 = len(df3['girls2'])
n3_list = list(df3['morekids'])


al2_mar = ['workedm','weeksm1','hourswm','incomem','famincl', 'workedd','weeksd1','hourswd','incomed','nonmomil']
for i in al2_mar:
    ln = df3[i]
    listlist=[]
    for j in range(n3):
        if n3_list[j] ==1:
            listlist.append((1/p)*ln[j])
        else:
            listlist.append((ln[j]/(p-1)))
    df3[i+"pp"] = listlist
    print(i)


from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import numpy as np
from sklearn.tree import export_graphviz 
import collections
import pydotplus


from sklearn.tree import DecisionTreeRegressor   
regressor = DecisionTreeRegressor(random_state = 0, max_depth=2)  
y1 = ['workedmpp','weeksm1pp','hourswmpp','incomempp','faminclpp']
y2 = ['workedmpp','weeksm1pp','hourswmpp','incomempp','faminclpp', 'workeddpp','weeksd1pp','hourswdpp','incomedpp','nonmomilpp']
features = np.asarray(['agem1','agefstm','educm','blackm','whitem','hispm','othracem'])


for m in y2:
    regressor.fit(np.asarray(df3[['agem1','agefstm','educm','blackm','whitem','hispm','othracem']]), np.asarray(df3[m]))
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

