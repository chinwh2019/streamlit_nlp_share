import base64
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer



class DataCleaner:
    def __init__(self, df):
        self.df = df

    def remove_missing_values(self, columns=None, threshold=0.5):
        """
        Remove rows with missing values. If the columns parameter is specified, only remove rows with missing values in those
        columns. The threshold parameter specifies the percentage of missing values allowed in a row.
        """
        if columns is not None:
            self.df = self.df.dropna(subset=columns, thresh=int(threshold * len(self.df)))
        else:
            self.df = self.df.dropna(thresh=int(threshold * len(self.df)))

    def remove_duplicates(self, columns=None):
        """
        Remove duplicate rows. If the columns parameter is specified, only consider those columns when identifying duplicates.
        """
        self.df = self.df.drop_duplicates(subset=columns)

    def remove_outliers(self, column, threshold=3):
        """
        Remove rows with outlier values in the specified column. The threshold parameter specifies the number of standard deviations
        from the mean that a value must be to be considered an outlier.
        """
        z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        self.df = self.df[abs(z_scores) < threshold]

    def impute_missing_values(self):
        # Create separate SimpleImputer objects for numeric and categorical variables
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # Identify numeric and categorical columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        categorical_columns = self.df.select_dtypes(exclude=[np.number]).columns

        # Impute missing values for numeric columns
        self.df[numeric_columns] = numeric_imputer.fit_transform(self.df[numeric_columns])

        # Impute missing values for categorical columns
        self.df[categorical_columns] = categorical_imputer.fit_transform(self.df[categorical_columns])

    def encode_categorical_variables(self, columns=None):
        """
        Encode categorical variables using one-hot encoding. If the columns parameter is specified, only encode those columns.
        """
        self.df = pd.get_dummies(self.df, columns=columns)

    def scale_features(self, feature_column, method='standard'):
        """
        Scale the specified feature column using the specified scaling method. The supported methods are 'standard' (standardization),
        'minmax' (min-max scaling), and 'robust' (robust scaling).
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f'Invalid scaling method: {method}')
        self.df[feature_column] = scaler.fit_transform(self.df[feature_column].values.reshape(-1, 1))

    def engineer_features(self, features):
        """
        Perform feature engineering, such as creating new features, combining existing features, or dropping irrelevant features.
        """
        for feature in features:
            if self.df[feature].dtype == 'int64' or self.df[feature].dtype == 'float64':
                self.df[f'new_{feature}'] = self.df[feature] + 1
            elif self.df[feature].dtype == 'object':
                self.df[f'new_{feature}'] = self.df[feature] + '_new'

    def get_clean_data(self):
        return self.df


# Test the DataCleaner class with streamlit app
def main():
    # Create the Streamlit app
    st.title('Data Cleaning App')

    # Add a file upload widget for the user to upload their dataset
    uploaded_file = st.file_uploader('Upload your dataset', type=['csv'])

    # If a file has been uploaded, preprocess the data and display it in a table
    if uploaded_file is not None:
        # Read the dataset into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Create a DataCleaner object
        cleaner = DataCleaner(df)

        # Add a multiselect widget for selecting the columns
        all_columns = df.columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = list(set(all_columns) - set(numeric_columns))
        selected_columns = st.multiselect('Select columns for outlier removal and scaling', all_columns)

        # Add a multiselect widget for selecting the features to engineer
        feature_columns = list(set(all_columns) - set(selected_columns))
        selected_features = st.multiselect('Select features for engineering', feature_columns)

        # Add a button to trigger data cleaning
        if st.button('Clean Data'):
            # Perform some cleaning operations
            cleaner.remove_missing_values()
            cleaner.remove_duplicates()
            for col in selected_columns:
                if col in numeric_columns:
                    cleaner.remove_outliers(col)
                    cleaner.scale_features(col)
                elif col in categorical_columns:
                    cleaner.encode_categorical_variables([col])
            cleaner.impute_missing_values()

            # Engineer the selected features
            cleaner.engineer_features(selected_features)

            # Get the cleaned data
            cleaned_df = cleaner.get_clean_data()

            # Display the cleaned data in a table
            st.write(cleaned_df)

            # Add a button to download the cleaned dataset as a CSV file
            csv = cleaned_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            button_label = 'Download cleaned data CSV File'
            button_url = f'data:file/csv;base64,{b64}'
            st.download_button(label=button_label, data=b64, file_name='cleaned_data.csv', mime='text/csv')


if __name__ == '__main__':
    main()
