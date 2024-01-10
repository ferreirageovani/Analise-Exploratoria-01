import pandas as pd 
import sweetviz as sv 
import numpy as np 

iris = pd.read_csv('datasets\\Iris.csv')

#my_report = sv.analyze(datafreme) ---> cria o report chamado my_report
#my_report.show_html()

msk = np.random.rand(len(iris)) < 0.8 # estipula um percentual da base para comparar
train = iris[msk]
test = iris[~msk]

train.head()
test.head()

#my_report = sv.compare([train, 'training set'], [test, 'testing set']) --->  compara duas partes da mesma base
#my_report.show_html()

my_report = sv.compare_intra(iris, iris['Species']=='Iris-setosa', ['Iris-setosa','Outros'])
my_report.show_html()


