# Domestic Indian Airline Ticket Price Analysis

## Project Overview
This project analyzes domestic Indian airline ticket prices to identify factors influencing fare costs and develop predictive models for future price determination. We focused on various factors like airline companies, class of travel, number of stops, duration, and days left until departure.

## Dataset
The dataset contains 30,153 observations with details such as airline name, class, departure and arrival times, duration, stops, and price. Data was sourced from Kaggle and encompasses both economy and business class fares.

- [Dataset Source](https://www.kaggle.com/code/borandabak/flight-price-data-analysis/data)

## Analysis and Results
Our exploratory data analysis (EDA) revealed significant variables like travel class, duration, and stops that impact ticket prices. We employed linear regression, decision trees, and KNN regression models to predict the prices, with decision trees showing the most promise.

## Key Findings
- Travel class significantly affects ticket prices.
- The number of stops and flight duration are crucial in price determination.
- Decision tree models outperformed linear and KNN regression in predicting ticket prices.

## Usage
To replicate our analysis or conduct further research, you can download the dataset from the link provided and follow the methodologies outlined in our technical documents.

## Repository Structure
- `Research Proposal.pdf`: Contains the initial research proposal and methodology.
- `Research Summary Report.pdf`: Offers a detailed analysis of our findings and methodologies.
- `Final Code and technical Part 1.py`: Python script for data cleaning, preprocessing, and EDA.
- `Final Code and technical Part 2.py`: Python script for modeling and further analysis.

## Requirements
- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels

## Installation
To set up your environment to run the code, you need to install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
