# Loan-approval-prediction

Predicting Loan Approval Using KNN and Decision Trees
Understanding the Problem Statement
The business case for this project is to develop a model that can predict loan approval for individuals. By using previous data on loan decisions made by the company, the goal is to create a machine learning model that can automate and simplify the loan approval process, reducing risk and improving decision-making efficiency. In this project, we will develop a model to predict who is eligible for a loan in order to reduce the risk associated with the decision process and to modify the typical loan approval process into a much easier one. Moreover, we will make use of previous data of loan decisions made by the company and with the help of various data mining techniques, we will develop a loan approval decision predicting model which can draw decisions for each individual based on the information provided by them.

The steps involved in solving this problem include:

- Data exploration and preprocessing: Analyze the dataset, handle missing values, convert categorical variables into numeric form, and normalize or standardize numerical variables if required.
- Splitting the data: Divide the dataset into features (X) and the target variable (y), and further split it into training and test sets to assess model performance.
- Model training: Build and train a KNN classifier and a Decision Tree classifier on the training data.
- Model evaluation: Make predictions using the trained models on the test set and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.
- Model comparison: Compare the performance of the KNN and Decision Tree models to determine which one performs better for loan approval prediction

I have included an html file for you in order to see the project without executing the code.
You can also run the project code files by following the following steps:

# Instructions

You would need to have Python installed on your machine. The project code is written in Python, and we will be using libraries like pandas, numpy, sklearn, matplotlib, seaborn. You can download Python from here.

1. Clone the project repository: Clone or download the project repository to your local machine using the command:


    git clone <repo_url>
    Replace <repo_url> with the URL of project Git repository.

2. Navigate to the project directory: Change your current directory to the directory of the project where the requirements file is located.

    cd <project-directory>
    Replace <project-directory> with the path of project directory.

3. Create a virtual environment (Optional): Creating a virtual environment is a good practice to keep the project's dependencies isolated from other projects. Use the following command to create a new virtual environment.

    python3 -m venv env
    To activate the virtual environment, use:


source env/bin/activate  # For Unix or MacOS
or

.\env\Scripts\activate  # For Windows

4. Install the project dependencies: The project dependencies are listed in the requirements.txt file. You can install these using the following command:
Copy code
pip install -r requirements.txt
This command will install all the Python dependencies required for the project.

Run the project: After the successful installation of all dependencies, you can run the project. You can execute the project with running the following:

python Predicting Loan Approval Using KNN and Decision Trees.py

