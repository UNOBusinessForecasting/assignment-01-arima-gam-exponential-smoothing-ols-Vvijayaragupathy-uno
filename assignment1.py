import statsmodels.formula.api as smf
import pandas as pd

# Load the training dataset
train_df = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

# Preview the training data
print(train_df.head())

# Explore the data
train_df.head()


ols_model = smf.ols("trips ~ hour", data=train_df)
trained_model = ols_model.fit()


print(trained_model.summary())


test_df = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_df = test_df[['hour']]  # Only keep the 'hour' column
print(test_df.head(10))  # Preview the test data

test_predictions = trained_model.predict(test_df)


print(test_predictions)


test_df.to_csv('predicted_trips.csv', index=False)

print("Predictions saved to 'predicted_trips.csv'")
