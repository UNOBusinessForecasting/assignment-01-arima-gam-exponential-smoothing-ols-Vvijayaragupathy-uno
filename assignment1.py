import statsmodels.formula.api as smf
import pandas as pd

# Load the training data
train_data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

# Explore the data
train_data.head()

# Fit the OLS model using 'hour' as the independent variable
ols_model = smf.ols("trips ~ hour", data=train_data)

# Fit the model
fitted_model = ols_model.fit()

# Display the summary of the fitted model
fitted_model.summary()
print(fitted_model.summary())

# Load the test data
test_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_df = test_df[['hour']]  # Select only the 'hour' column
test_df.head(10)

# Use the fitted model to predict 'trips' based on the 'hour' from the test data
predicted_trips = fitted_model.predict(test_df)

# Output the predictions
print(predicted_trips)
