# :bank: Loan Approval Prediction :bar_chart:

This project aims to develop a model that can predict loan approval for individuals. The goal is to automate and simplify the loan approval process, reducing risk and improving decision-making efficiency. 


## :clipboard: Problem Statement

Develop a model to predict loan eligibility for individuals. Use various data mining techniques to build a loan approval decision predicting model, which can make decisions based on the information provided by the individuals.

## :hammer_and_wrench: Steps Involved

1. **:mag_right: Data Exploration and Preprocessing**: Analyze the dataset, handle missing values, convert categorical variables into numeric form, and normalize or standardize numerical variables if required.
   
2. **:scissors: Splitting the Data**: Divide the dataset into features (X) and the target variable (y). Further split it into training and test sets to assess model performance.
   
3. **:gear: Model Training**: Build and train a KNN classifier and a Decision Tree classifier on the training data.
   
4. **:chart_with_upwards_trend: Model Evaluation**: Make predictions using the trained models on the test set and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.
   
5. **:balance_scale: Model Comparison**: Compare the performance of the KNN and Decision Tree models to determine which one performs better for loan approval prediction.

## :package: Instructions for Running the Project

1. **Python**: You need to have Python installed on your machine. You can download Python from [here](https://www.python.org/downloads/).


2. Clone this repo:

```
git clone https://github.com/Minakoaino/Loan-approval-prediction.git
```

3. **Navigate to the Project Directory**: Change your current directory to the directory of the project where the `requirements.txt` file is located:
 
 ```
cd Loan-approval-prediction
 ```

4. **Virtual Environment** (Optional): Creating a virtual environment is a good practice to keep the project's dependencies isolated from other projects:

- Create a new virtual environment:
  
  ```
  python3 -m venv env
  ```
  
- Activate the virtual environment:
  
  - For Unix or MacOS:
    
    ```
    source env/bin/activate
    ```
    
  - For Windows:
    
    ```
    .\env\Scripts\activate
    ```

5. **Install the Project Dependencies**: The project dependencies are listed in the `requirements.txt` file. You can install these using the following command:


