# COMP 472 Naive Bayes Bag of Words (aiproj3)
https://github.com/prich28/aiproj3

Naive Bayes Bag of Words

1. Clone and download the project into a local folder.

2. Create a conda environment with python3 installed (activate that environment

  `conda create -n my_env python=3.8`
  
  `conda activate my_env`
  
3. Navigate to the folder where the project is downloaded. Inside the `aiproj3` folder (where main.py is located) place the TSV training file and TSV test file.

**NOTE: Both the training and testing file must be tsv. The first line may or may not contain the column headers (both will work). If it contains the header, the header must be the same as provided to us in the covid_training.tsv file. Either, the first cell must contain the string: `tweet_id`, or the second cell must contain the string: `text` or the third cell must contain the string: `q1_label`. If one of these conditions are not met, it will assume that the first line is a tweet and not work properly. In any file, the data in the first column must be the tweet id, the second column, the text of the tweet and the third column, the text yes or no (class). Any other formats will not work.

4. In a console/terminal (with the conda environment activated), navigate to the project folder where the main.py file is located. Run the command:
 `python3 main.py`
 
5. Follow the on-screen instructions (type in the name of the training file then test file with extension

6. The trace output and eval output files for both the original vocabulary and the filtered vocabulary will appear in the same folder as the `main.py` file.
