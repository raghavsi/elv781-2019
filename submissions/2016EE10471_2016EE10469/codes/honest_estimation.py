nn=200000
from sklearn.tree import DecisionTreeRegressor   
from sklearn.metrics import mean_squared_error

regressor = DecisionTreeRegressor(random_state = 0, max_depth=3)  

y = ['workedm','weeksm1','hourswm','incomem','famincl', 'workedd','weeksd1','hourswd','incomed','nonmomil']
features = np.asarray(['agem1','agefstm','educm','blackm','whitem','hispm','othracem'])

for m in y:
    regressor.fit(np.asarray(df2[['agem1','agefstm','educm','blackm','whitem','hispm','othracem']][0:nn]), np.asarray(df2[m][0:nn]))
    yp=regressor.predict(df2[['agem1','agefstm','educm','blackm','whitem','hispm','othracem']][nn:])
    y_t= np.asarray(df2[m][nn:])
    print(m,mean_squared_error(yp,y_t)/(np.mean(y_t**2)))