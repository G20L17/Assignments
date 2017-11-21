from util2 import Arff2Skl
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus as pydot

cvt = Arff2Skl('contact-lenses.arff')
label = cvt.meta.names()[-1]
X, y = cvt.transform(label)
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X,y)

features = ['age=pre-presbyopic','age=presbyopic','age=young' ,'spectacle-prescrip=myope',
            'spectacle-prescrip','astigmatism=yes','astigmatism=no','tear-prod-rate=normal',
            'tear-prod-rate=reduced']
classes = ['hard','none','soft']
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, rounded=True, feature_names=features,
                     class_names=classes,
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('Q1.2_Tree_Graph.pdf')