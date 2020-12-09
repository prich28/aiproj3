import math
from naive_bayes.nb_tweet_result import NBTweetResult


class NaiveBayes:
    def __init__(self, training_data):
        self.training_data = training_data
        self.classes = ['yes', 'no']
        self.vocabulary = []
        self.p_word_class = {}
        self.priors = {}
        self.smoothing_value = 0.01

    # Get vocabulary Go through all the tweets, record every unique word
    def create_vocabulary(self):
        for item in self.training_data:
            list_of_words = item[1].split()
            for word in list_of_words:
                lower_case_word = word.lower()
                if lower_case_word not in self.vocabulary:
                    self.vocabulary.append(lower_case_word)

        print("Original vocabulary count: " + str(len(self.vocabulary)))

    # Creates a vocabulary only with words that appear more than once in training set
    def create_filtered_vocabulary(self):
        all_words = []
        for item in self.training_data:
            list_of_words = item[1].split()
            for word in list_of_words:
                lower_case_word = word.lower()
                all_words.append(lower_case_word)

        for word in all_words:
            if word not in self.vocabulary:
                occurrence = all_words.count(word)
                if occurrence > 1:
                    self.vocabulary.append(word)

        print("Filtered vocabulary count: " + str(len(self.vocabulary)))

    # Sets the prior probabilities of the classes
    def set_priors(self):
        yes_count = 0
        no_count = 0

        for item in self.training_data:
            if item[2] == "yes":
                yes_count += 1
            if item[2] == "no":
                no_count += 1

        total_count = float(len(self.training_data))

        self.priors["yes"] = math.log10(float(yes_count)/total_count)
        self.priors["no"] = math.log10(float(no_count)/total_count)

    # Sets each vocab word probability by class
    def set_word_probability_by_class(self):
        # Can have duplicates (only words in the vocabulary)
        yes = []
        no = []

        # Separate all words into each class (does not add words not in vocabulary)
        for item in self.training_data:
            tweet = item[1]
            if item[2] == "yes":
                list_of_words = tweet.split()
                for word in list_of_words:
                    lower_case_word = word.lower()
                    if lower_case_word in self.vocabulary:
                        yes.append(lower_case_word)
            if item[2] == "no":
                list_of_words = tweet.split()
                for word in list_of_words:
                    lower_case_word = word.lower()
                    if lower_case_word in self.vocabulary:
                        no.append(lower_case_word)

        # Including all duplicates of words (but NOT words not contained in the vocabulary)
        word_count_yes = float(len(yes))
        word_count_no = float(len(no))

        self.p_word_class["yes"] = {}
        self.p_word_class["no"] = {}

        for word in self.vocabulary:
            # Smoothing denominator value
            smoothed_vocab_size = float(len(self.vocabulary))*self.smoothing_value

            freq_in_yes = float(yes.count(word))
            smoothed_freq_yes = freq_in_yes + self.smoothing_value
            self.p_word_class["yes"][word] = math.log10(smoothed_freq_yes/(word_count_yes + smoothed_vocab_size))

            freq_in_no = float(no.count(word))
            smoothed_freq_no = freq_in_no + self.smoothing_value
            self.p_word_class["no"][word] = math.log10(smoothed_freq_no/(word_count_no + smoothed_vocab_size))

    # Categorizes a tweet by class based on naive bayes filtering
    def tweet_tester(self, test_data):
        results = []
        for datum in test_data:
            tweet_number = datum[0]
            tweet = datum[1]
            real_value = datum[2]

            yes_score = self.priors["yes"]
            no_score = self.priors["no"]

            list_of_words = tweet.split()
            for word in list_of_words:
                lower_case_word = word.lower()
                if lower_case_word in self.vocabulary:
                    yes_score += self.p_word_class["yes"][lower_case_word]
                    no_score += self.p_word_class["no"][lower_case_word]

            prediction = "yes" if (yes_score > no_score) else "no"
            score = yes_score if (prediction == "yes") else no_score

            result = NBTweetResult(tweet_number, prediction, score, real_value)
            results.append(result)

        return results
