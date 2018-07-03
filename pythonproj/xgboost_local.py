from __future__ import print_function
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pickle
import xgboost as xgb
import pandas as pd



def get_model_from_data(data_path,model_path):

    # file_name = '/home/avinash/platform/image-processing/capabilities/pg_dataset.pkl'
    file_name = data_path
    data = pickle.load(open(file_name,"rb"))

    features = data['fv_list']
    labels = data['label_list']
    import pdb; pdb.set_trace()
    

    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
        np.array(labels), test_size=0.5, random_state=42, stratify=labels)
    weights = np.zeros(len(trainLabels))
    one_count = sum( trainLabels )
    zero_count = len(trainLabels) - one_count
    min_ = min( one_count, zero_count )
    if( min_ == 0 ):
        print( "Invalid training set! " )
        return -1

    weights[trainLabels == 0 ] = 1.0*one_count/min(one_count, zero_count)
    weights[trainLabels == 1] = 1.0*zero_count/min(one_count, zero_count)


    dtrain = xgb.DMatrix(trainData,label=trainLabels,weight=weights)
    dtest = xgb.DMatrix(testData)

    ## Play with threshold 
    params = {
        'objective':'binary:logistic',
        'max_depth':1,
        'silent':1,
        'eta':1
    }

    num_rounds = 15

    bst = xgb.train(params,dtrain,num_rounds)
    y_test_preds = (bst.predict(dtest)>0.5).astype('int') ## Play with threshold 

    # filename = 'finalized_model.sav'
    filename = model_path
    pickle.dump(bst, open(filename, 'wb'))

    print(pd.crosstab(pd.Series(testLabels,name='Actual'),
                pd.Series(y_test_preds,name='Predicted'),
                margins = True))

    print(classification_report(testLabels, y_test_preds))
    print(confusion_matrix(testLabels, y_test_preds))





get_model_from_data('kv_dataset.pkl','xgboost_kv_model.sav')
