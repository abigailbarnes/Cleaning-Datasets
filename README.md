# Cleaning-Datasets
Task 1: Exploration: Preparing Data for Public Release (34 points)
It's tax season! Imagine that you have been hired as an IT consultant for the Internal Revenue Service (IRS), who have assembled what they believe to be the most relevant data from a random sample of 500,000 Americans' tax returns. To support research into the socioeconomics of Americans, they want you to take the data they have sampled from tax returns and prepare it for public release. Of course, there might be some private information in this file, so they want you to pre-process the file in a way that maintains as much utility of the data as possible while minimizing the risks of releasing this data.

You can download the raw CSV file your supervisors have provided here.

Write-up Question 1 (5 points): Describe what you feel are the risks of releasing the data in the original form (i.e., the raw file provided to you above).

Write code (in whatever language you want) to pre-process and prepare this data for public release. You can use whatever strategies you want. Note that this is an exploration task, so you are graded based on your effort, not on your ultimate success in being ethical and responsible. The file you have prepared for public release should be named release.csv.

Write-up Question 2 (5 points): Describe your strategy and approach for creating a data release that maintains as much utility as possible while minimizing the risks described above.

Exploration: Building an ML Model (41 points)
We'll first get hands-on experience building a machine learning (ML) model. For those of you who have taken an applied ML class, this section will mostly be a review of things you have hopefully done before and should be pretty quick. If you haven't previously built an ML model, this task will hopefully give you hands-on experience with a process that has become a fairly core skill for a modern computer scientist.

To do so, we'll start with a dataset loan_data.csv that contains data about loan applications and whether or not they were approved. The dataset includes 1,307 rows of data about the loan applicants --- their race, their gender, the date of the application, their ZIP code, their income, the type of loan, the term of the loan (in months), the loan's interest rate, the principal (the amount of the loan), whether the loan was ultimately approved, a column labeled adj_bls_2 (we're not sure what this is either!), and the ID number of the transaction.

In this task, your goal is to build a binary classifier --- an ML model that outputs "yes" or "no" for whether the loan should be approved --- based on the dataset. We are intentionally leaving exactly how you do so open-ended.

We highly recommend completing this task in Python, reading the data into a Pandas dataframe Links to an external site.and using the scikit learn Links to an external site.(sklearn) library to train your binary classifier. Our course staff can provide by far the most help if you use those libraries. If you'd rather do something else, you're welcome to do so, but you won't get much support from us.

Task 1 Step 1: Data Cleaning
Your first step after reading the data in will be to "clean" the data to correct any errors and inconsistencies. Write code that does this.

Write-up Question 1 (5 points): Describe the steps you took in data cleaning: what types of data errors/issues you looked for, what you found, and what steps you took to address these errors.

Task 1 Step 2: Data Preparation
Next, you'll need to prepare the data for building a machine learning model. In addition to making sure that the numeric data is properly encoded (and you'll have to decide whether or not the numeric data should actually be treated as numeric), there are several categorical variables in this dataset. For example, race contains categories, not numbers. Because common ML libraries don't directly handle categorical variables, you'll need to use dummy-coding to create columns with binary values to represent the different categories. Luckily, there is already a pandas function that does this for you: get_dummies Links to an external site..

If you have never done this sort of dummy-coding before, please feel encouraged to search for and follow an online tutorial, of course properly attributing the tutorial you followed in Question 0 at the top of your problem set.

At this point, you will have a dataframe containing columns of either just numbers (for numerical variables) or 1s and 0s (dummy-coded categorical variables, which may expand into multiple columns).

You'll also need to store the label (whether or not the loan was approved) in its own variable whose length is the same as the number of rows in the aforementioned dataframe.

Task 1 Step 3: Model Training
Next, you will create an ML model using scikit learn Links to an external site.(sklearn). In particular, you will be building a binary classifier that tries to model whether a data subject is likely at risk for a stroke. First, split the data into a training set and a test set. Please use some fraction of your data for the training set. Second, train some sort of classifier using your training set. Third, evaluate that classifier on your test set. Fourth, print out the model's accuracy, precision, recall, and F1 score based on your test set. Fifth, also create a confusion matrix Links to an external site..

If you have never performed any of these steps, I highly recommend this tutorial Links to an external site.for building an SVM model. It's on the shorter side, yet provides most of what you need to know. Follow the "Splitting Data" (for the train-test split), "Generating Model" (for training the classifier), and "Evaluating the Model" sections in particular. The early parts of the aforementioned tutorial succinctly explain what you're doing conceptually with the sample code, though it's focused around a sample dataset that's different than the one you are using. To augment that tutorial's last section, also take a look at the documentation for scikit learn's metrics package Links to an external site.. To create a confusion matrix, I'd recommend this sample code Links to an external site.from the same library. That last example is taken from a tutorial that I also found pretty good as a whole, though it goes into a lot of detail on tasks beyond those we're asking you to complete.

Task 1 Step 4: Selection of a Final Model
Ultimately, we want you to try out a couple of different models, different ways of pre-processing the data, and ultimately select a final model that you would deploy as an automated system for approving/declining loan applications. At the end of your code, the final model you have chosen should be obvious to us (e.g., stored in a variable called finalModel and/or indicated as the final model in a comment in your code). We will be returning to this final model and re-evaluating it in a future week.

Write-up Question 2 (5 points): Briefly describe your process of selecting a model. What did you try (this should also be obvious from your code!), and what led you to select the final model you did? Note that this process could involve both actions on the data itself, as well as the model. Your answer should cover both.

Write-up Question 3 (3 points): Briefly describe what predictive variables you chose for your final model, as well as the model architecture (e.g., SVM, Decision Tree). The predictive variables may be columns from the original dataset (unmodified), post-processed information, etc.

Write-up Question 4 (6 points): Briefly describe your final model's performance. You should present at least the accuracy, precision, and recall, but you are encouraged to look up other possible metrics to include. For each metric, in addition to the formal term (e.g., "precision"), describe what this metric means less formally in the context of approving/declining loans.

Write-up Question 5 (4 points): As an ethical computer scientist, how comfortable would you be deploying this model to make automated decisions about approving/declining loans? Why do you feel this way?

Task 2: Exploration: Data Subject Access Requests (25 points)
As we'll discuss in lecture during a future week, recent privacy laws in the US and in the EU have strengthened data access rights, specifically the ability for users to make a data subject access request and "download their data" from different companies. You can do so from most companies with which you have online accounts, including major players like Google Links to an external site., Facebook Links to an external site., Instagram Links to an external site., Twitter Links to an external site., Snapchat Links to an external site., Doordash Links to an external site., Spotify Links to an external site., Amazon Links to an external site., and many others. You can also download data from companies that you might not realize are collecting data about you, as a New York Times article discussed Links to an external site..

Request your data from at least five different companies that you expect will have data about you. While you are allowed to use the companies listed above, we encourage you to request your data from at least some companies not on this list since we are very curious what you find from different companies. You can usually find an appropriate request page for a company by searching for {"download", "request"} {"my", "your"} {"data", "personal information"}, though some companies require you to email/call them, or even to write a letter like it's still 1998. Note that some companies respond to data subject access requests within a few hours, others take days, and some take weeks or longer.

Write-up Question 6 (5 points): List the 5+ companies from which you've requested data. You are encouraged to request your data from more than five companies since, as mentioned above, some companies take a while to respond.

Do not upload your data downloads to us! They contain potentially sensitive information, and we don't want them!

Write-up Question 7 (15 points total; 3 points per company): Succinctly characterize the process by which you requested and received your data, as well as the contents and format of the data download that the company provided you if you have already received it. If you have not already received your data by the time you submit the assignment, just say so for those companies. You may answer this question in a table, in bulleted lists, or using whatever format makes the most sense to you. At the minimum, make sure your answer covers: (i) the process by which you made the request, including the URL if applicable; (ii) how the company authenticated that you are the rightful data subject, if at all; (iii) how the response was provided to you, such as via a download link emailed to you; (iv) the format and structure of the "data download" if you have received it already; (v) what broad types of information are contained in the data download if you have received it already; and (vi) what time period the data download covers if you have received it already. For each point, don't stress about being fully comprehensive, but give the broad idea. Please also feel encouraged to note anything you find interesting about each company's download. If you received data from more than five companies, pick five for this part of the assignment.

Write-up Question 8 (5 points): Characterize how companies might differ from each other in interpreting requirements to respond to data subject access requests. Specifically, companies are required to provide the "personal data" associated with the person making the request. Briefly describe any data you might have expected a company might collect about users that none of the companies included in their data downloads. If you haven't yet received data from any companies, instead list the types of data that companies plausibly collect and retain that may be associated with a particular user or account.
