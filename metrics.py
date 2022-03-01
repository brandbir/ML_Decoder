import numpy as np
import pandas as pd
import ast

def prediction_accuracy(list_actual, list_pred):
    """
    This function is used to evaluate the accuracy obtained after predicting prepositions.
    This is achieved by checking whether any of the predicted prepositions is in the actual preposition list.
    :param list_prep_actual: list containing actual prepositions per each instance
    :param list_prep_pred: list containing predicted prepositions per each instance
    :return: accuracies list containing 1s and 0s in relation to the prediction per each instance
    """
    assert len(list_actual) == len(list_pred), 'Lists should have equal length'
    accuracies = []

    for idx, prep_pred in enumerate(list_pred):
        prep_actual_str = list_actual[idx]
        prep_actual_list = [x.strip() for x in prep_actual_str.split(',')]

        acc = 0
        for pred in prep_pred:
            if pred in prep_actual_list:
                acc = 1
                break

        accuracies += [acc]

    return np.array(accuracies).mean()


def intersection(actual, predicted):
    """
    This function computes the intersection between the ground truth list
    and the predicted list
    :param actual: list containing ground truths
    :param predicted: list containing predictions
    :return : intersection between actual and predicted lists"""
    return set(actual).intersection(set(predicted))


def union(actual, predicted):
    """
    This function computes the union between the ground truth list
    and the predicted list
    :param actual: list containing ground truths 
    :param predicted: list containing predictions
    :return : un between actual and predicted lists"""
    return set(actual).union(set(predicted))


def accuracy(actual, predicted):
    """
    This function computes the accuracy between each ground truth and predicted set
    :param actual: list of ground truth lists
    :param predicted: list of predicted lists
    :return: accuracy per each pair in a list
    """
    assert len(actual) == len(predicted), "List should have equal length"
    accuracy_list = []

    for idx, a in enumerate(actual):
        a = ast.literal_eval(a)
        p = ast.literal_eval(predicted[idx])

        if type(a) is str:
            a = a.split(',')

        # set is used to remove duplicates found in the ground truth set
        a = list(set(a))

        assert type(a) == type(p) == list, "Actual and predicted should be in list."
        acc = len(intersection(a, p)) / float(len(union(a, p)))
        accuracy_list += [acc]

    assert 0 <= max(accuracy_list) <= 1, "Max accuracy rate should be less than or equal to 1."
    assert 0 <= min(accuracy_list) <= 1, "Min accuracy rate should be less than or equal to 1."

    return accuracy_list


def recall(actual, predicted):
    """
    This function computes the recall between each ground truth and predicted set
    :param actual: list of ground truth lists
    :param predicted: list of predicted lists
    :return: recall per each pair in a list
    """
    assert len(actual) == len(predicted), "List should have equal length"
    recall_list = []

    for idx, a in enumerate(actual):

        a = ast.literal_eval(a)
        p = ast.literal_eval(predicted[idx])

        if type(a) is str:
            a = a.split(',')

        # set is used to remove duplicate found in the ground truth  set
        a = list(set(a))
    
        assert type(a) == type(p) == list, "Actual and predicted should be in list."
        r = len(intersection(a, p)) / float(len(a))
        recall_list += [r]

    assert 0 <= max(recall_list) <= 1, "Max recall rate should be less than or equal to 1."
    assert 0 <= min(recall_list) <= 1, "Min recall should be less than or equal to 1."

    return recall_list


def precision(actual, predicted):
    """
    This function computes the precision between each ground truth and predicted set
    :param actual: list of ground truth lists
    :param predicted: list of predicted lists
    :return: precision per each pair in a list
    """
    assert len(actual) == len(predicted), "Lists should have equal length."
    precision_list = []

    for idx, a in enumerate(actual):
        a = ast.literal_eval(a)
        p = ast.literal_eval(predicted[idx])

        if type(a) is str:
            a = a.split(',')

        # set is used to remove duplicate found in the ground truth  set
        a = list(set(a))
        
        assert type(a) == type(p) == list, "Actual and predicted should be in list."

        if len(p) == 0:
            prc = 0
        else:
            prc = len(intersection(a, p)) / float(len(p))

        precision_list += [prc]

    assert 0 <= max(precision_list) <= 1, "Max precision should be less than or equal to 1."
    assert 0 <= min(precision_list) <= 1, "Min precision should be less than or equal to 1."

    return precision_list


def fscore(precisions, recalls):
    f_score_numerator = 2 * (precisions * recalls)
    f_score_denominator = precisions + recalls
    f_score = f_score_numerator / f_score_denominator
    f_score = np.nan_to_num(f_score)

    return f_score


def generate_metric_per_label(actuals, predicted, metric='recall'):
    totals = {}
    metrics = {} # recalls or precisions per label

    if metric == 'recall':
        list_1 = actuals
        list_2 = predicted

    elif metric == 'precision':
        list_1 = predicted
        list_2 = actuals
    
    
    assert len(actuals) == len(predicted), 'length of actuals and predicted should be equal.'
    
    for idx, a in enumerate(list_1):
        a = ast.literal_eval(a)
        #p = predicted[idx]
        p = ast.literal_eval(list_2[idx])

        if type(a) is str:
            a = a.split(',')

        # set is used to remove duplicate found in the ground truth  set
        a = list(set(a))
        
        assert type(a) == type(p) == list, "Actual and predicted should be in list."
        
        for ai in a:
            if ai in totals.keys():
                totals[ai] += 1
            else:
                totals[ai] = 1

            if ai in p:
                if ai in metrics.keys():
                    metrics[ai] += 1
                else:
                    metrics[ai] = 1

    for tk in totals.keys():
        if tk not in metrics.keys():
            metrics[tk] = 0

        metrics[tk] = metrics[tk]/float(totals[tk])
        
    
    return metrics


def avg_metric_per_label(metrics_per_label):
    avg_metric = 0
    for key in metrics_per_label:
        avg_metric += metrics_per_label[key]
    avg_r_prep = avg_metric / len(metrics_per_label)

    return avg_r_prep


def cardinality(preps):
    count = 0
    for p in preps:
        count += len(p)

    return count / len(preps)

def from_dict_to_pd(dict, value_col):
    cols = ['keyword', value_col]
    df = pd.DataFrame(columns=cols)
    
    for k in dict.keys():
        df = df.append(pd.DataFrame([[k, np.round(dict[k],3)]], columns=cols))

    return df.reset_index(drop=True)


def main():
    predictions_file_name = 'ml_predictions_test'
    predictions = pd.read_csv(predictions_file_name + '.csv')
    print('Computing metrics on', len(predictions), 'instances')

    actuals_classes = predictions['actual_classes'].values
    pred_classes = predictions['pred_classes'].values
    
    accuracy_list = accuracy(actuals_classes, pred_classes)
    avg_accuracy = np.array(accuracy_list).mean().round(4)

    precision_list = precision(actuals_classes, pred_classes)
    avg_precision = np.array(precision_list).mean().round(4)

    recall_list = recall(actuals_classes, pred_classes)
    avg_recall = np.array(recall_list).mean().round(4)
    
    fscore_list = fscore(np.array(precision_list), np.array(recall_list))
    avg_fscore = np.array(fscore_list).mean().round(4)

    recalls_per_label = generate_metric_per_label(actuals_classes, pred_classes, 'recall')
    avg_recall_per_label = round(avg_metric_per_label(recalls_per_label),4)

    precisions_per_label = generate_metric_per_label(actuals_classes, pred_classes, 'precision')
    avg_precision_per_label = round(avg_metric_per_label(precisions_per_label),4)

    print('A-Acc:', avg_accuracy, 'AP:', avg_precision, 'AR:', avg_recall, 'AF:', avg_fscore)
    print('Average recall per label:', avg_recall_per_label)
    print('Average precision per label:', avg_precision_per_label)

    predictions['accuracy'] = accuracy_list
    predictions['precision'] = precision_list
    predictions['recall'] = recall_list
    predictions['fscore'] = fscore_list

    predictions.to_csv(predictions_file_name + '_metrics.csv', index=False)

    df_recalls_per_label = from_dict_to_pd(recalls_per_label, 'recall')
    df_precisions_per_label = from_dict_to_pd(precisions_per_label, 'precision')

    df_metrics_per_label = df_recalls_per_label
    df_metrics_per_label['precision'] = df_precisions_per_label['precision']
    df_metrics_per_label = df_metrics_per_label.sort_values(by='precision', ascending=False)

    df_metrics_per_label.to_csv('ml_predictions_metrics_per_label.csv', index=False)

if __name__ == '__main__':
    main()
