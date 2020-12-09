import csv
from naive_bayes.naive_bayes import NaiveBayes


def import_data(file_name):
    data_array = []
    try:
        with open(file_name, ) as training_file:
            training_import_object = csv.reader(training_file, delimiter='\t')

            for data in training_import_object:
                data_array.append(data)
    except FileNotFoundError:
        print("     We cannot load the file '" + file_name + "'")
        print("     Make sure the file is in the correct folder, (with main.py)")
        print("     Make sure you have spelled the filename correctly")
        print("     The program will close, try running again!")
        exit(1)

    # In order to be able to import both the training set (with headers), pops the header if it exists
    if (data_array[0][0] == "tweet_id") or (data_array[0][1] == "text") or (data_array[0][1] == "q1_label"):
        data_array.pop(0)

    return data_array


def export_data(all_results, vocab_type):
    output_filename = "trace_NB-BOW-" + vocab_type + ".txt"
    with open(output_filename, "w") as output:
        for result in all_results:
            output.write(result.tweet_id + "  ")
            output.write(result.prediction + "  ")
            output.write("{:.2e}".format(result.score) + "  ")
            output.write(result.real_value + "  ")
            output.write(result.is_correct + "\n")


def arrange_data(imported_data):
    arranged_data = []
    for datum in imported_data:
        arranged_data.append([datum[0], datum[1], datum[2]])
    return arranged_data


def accuracy(all_results):
    correct_count = 0.0
    all_count = float(len(all_results))
    for tweet in all_results:
        if tweet.is_correct == "correct":
            correct_count += 1.0

    return correct_count / all_count


def precision(all_results, class_type):
    true_positives = 0.0
    tp_and_fp = 0.0

    for tweet in all_results:
        if (tweet.prediction == tweet.real_value) and (tweet.real_value == class_type):
            true_positives += 1.0
        if tweet.prediction == class_type:
            tp_and_fp += 1.0

    # For each class (ex. yes), all of the tweets labeled yes, that are yes, divided by all the tweets it labeled yes.
    return true_positives / tp_and_fp


def recall(all_results, class_type):
    true_positives = 0.0
    tp_and_fn = 0.0

    for tweet in all_results:
        if (tweet.prediction == tweet.real_value) and (tweet.real_value == class_type):
            true_positives += 1.0
        if tweet.real_value == class_type:
            tp_and_fn += 1.0

    # For each class (ex. yes), all of the tweets labeled yes, that are yes, divided by all the tweets that are yes.
    return true_positives / tp_and_fn


def f1_measure(p, r):
    return (2.0 * p * r) / (p + r)


def output_evaluation(all_results, vocab_type):
    output_filename = "eval_NB-BOW-" + vocab_type + ".txt"
    acc = accuracy(all_results)

    yes_precision = precision(all_results, "yes")
    yes_recall = recall(all_results, "yes")
    yes_f1 = f1_measure(yes_precision, yes_recall)

    no_precision = precision(all_results, "no")
    no_recall = recall(all_results, "no")
    no_f1 = f1_measure(no_precision, no_recall)

    with open(output_filename, "w") as output:
        output.write(str(round(acc, 4)) + "\n")
        output.write(str(round(yes_precision, 4)) + "  " + str(round(no_precision, 4)) + "\n")
        output.write(str(round(yes_recall, 4)) + "  " + str(round(no_recall, 4)) + "\n")
        output.write(str(round(yes_f1, 4)) + "  " + str(round(no_f1, 4)) + "\n")


training_file_name = input("Type the name of the training file (with extension): ")

test_file_name = input("Type the name of the test file (with extension): ")

# training_file_name = "covid_training.tsv"
# test_file_name = "covid_test_public.tsv"

training_data = arrange_data(import_data(training_file_name))
test_data = arrange_data(import_data(test_file_name))

# OV SECTION:
# Create NaiveBayes object with Original Vocabulary
nb_ov = NaiveBayes(training_data)

# Train NaiveBayes object
nb_ov.create_vocabulary()
nb_ov.set_priors()
nb_ov.set_word_probability_by_class()

# Run test set (output: array of tweet result objects)
ov_results = nb_ov.tweet_tester(test_data)

export_data(ov_results, "OV")
output_evaluation(ov_results, "OV")

# FV SECTION:
# Create NaiveBayes object with FilteredVocabulary
nb_fv = NaiveBayes(training_data)

# Train NaiveBayes object
nb_fv.create_filtered_vocabulary()
nb_fv.set_priors()
nb_fv.set_word_probability_by_class()

# Run test set (output: array of tweet result objects)
fv_results = nb_fv.tweet_tester(test_data)

export_data(fv_results, "FV")
output_evaluation(fv_results, "FV")

print("FINISHED!")
