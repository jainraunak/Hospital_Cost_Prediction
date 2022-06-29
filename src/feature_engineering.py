import numpy as np

def onehotencoding(data,feature_name,target_feature,high_freq_features):
    """
    Do one hot encoding of the feature = feature_name
    :param data: data including both train and test data
    :param feature_name: name of the feature to be one hot encoded
    :param target_feature: target feature to be used in one-hot encoding
    :param high_freq_features: features having high unique values
    :return: one hot encoded data
    """
    unique_list = data[feature_name].value_counts(sort=True,ascending=False).index
    n_arr = np.asarray(data[feature_name].values)
    target_arr = np.asarray(data[target_feature].values)
    for idx,value in enumerate(unique_list):
        data[feature_name+'_'+str(idx)] = np.where(n_arr == value,target_arr,0)
        if idx >= 20 and feature_name in high_freq_features:
            # Get one hot encoding of best 20 values only
            break
    return data

def multifeature_onehotencoding(data,feature_name1,feature_name2,target_feature):
    """
    Do multi feature one hot encoding of 2 features feature_name1 and feature_name2
    :param data: Data including both training and test data
    :param feature_name1: First feature to be used in one hot encoding
    :param feature_name2: Second feature to be used in one hot encoding
    :param target_feature: target feature to be used in one hot encoding
    :return: one hot encoded data
    """

    unique_list1 = data[feature_name1].value_counts(sort=True, ascending=False).index
    n_arr1 = np.asarray(data[feature_name1].values)
    unique_list2 = data[feature_name2].value_counts(sort=True, ascending=False).index
    n_arr2 = np.asarray(data[feature_name2].values)
    target_arr = np.asarray(data[target_feature].values)
    cnt = 0
    for idx1,value1 in enumerate(unique_list1):
        for idx2,value2 in enumerate(unique_list2):
            data[feature_name1 + '_' + feature_name2 + '_' + str(cnt)] = np.where(
                (n_arr1 == value1) & ((n_arr2 == value2)), target_arr, 0)
            cnt += 1
    return data

def feature_engineering(data):
    """
    Do feature engineering on the data.
    :param data: DataFrame storing training and test data
    :return: data after feature engineering
    """

    # Features to one hot encode
    features_to_one_hot_encode = ['Ethnicity', 'Type of Admission', 'Age Group', 'Operating Certificate Number',
                                  'Payment Typology 1', 'APR Severity of Illness Description',
                                  'APR Risk of Mortality', 'APR Medical Surgical Description',
                                  'APR MDC Description', 'CCS Procedure Description',
                                  'CCS Diagnosis Description', 'APR DRG Description']

    # High Frequency features
    high_frequency_features = ['CCS Procedure Description', 'CCS Diagnosis Description', 'APR DRG Description']

    for feature_name in features_to_one_hot_encode:
        data = onehotencoding(data=data, feature_name=feature_name, target_feature='Length of Stay',
                              high_freq_features=high_frequency_features)

    # Features to multi one hot encode
    multi_features_to_one_hot_encode = [('Health Service Area', 'Emergency Department Indicator'),
                                        ('Age Group', 'Emergency Department Indicator')]

    for feature_name1, feature_name2 in multi_features_to_one_hot_encode:
        data = multifeature_onehotencoding(data, feature_name1, feature_name2, target_feature='Length of Stay')

    # Features to drop
    features_to_drop = ['Ethnicity', 'Type of Admission', 'Age Group', 'Operating Certificate Number',
                        'Payment Typology 1', 'APR Severity of Illness Description',
                        'APR Risk of Mortality', 'APR Medical Surgical Description',
                        'Race', 'Gender', 'Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'APR DRG Code',
                        'APR MDC Code']

    data.drop(features_to_drop, inplace=True, axis=1)

    return data

