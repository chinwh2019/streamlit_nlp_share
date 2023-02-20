import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, data):
        self.data = data

    def summary_statistics(self):
        return self.data.describe()

    def data_visualization(self, plot_type='hist', column=None, **kwargs):
        plot_functions = {
            'hist': self.data.hist,
            'scatter': sns.scatterplot,
            'box': self.data.boxplot,
            'heatmap': sns.heatmap,
        }
        if plot_type not in plot_functions:
            raise ValueError("Unsupported plot type")
        if plot_type == 'scatter':
            x, y = column
            plot_kwargs = {'x': x, 'y': y, 'data': self.data, **kwargs}
        else:
            plot_kwargs = {'column': column, **kwargs}
        plot_function = plot_functions[plot_type]
        return plot_function(**plot_kwargs)

    def correlation_analysis(self, column1, column2):
        return self.data[[column1, column2]].corr()

    def dimensionality_reduction(self, method='pca', n_components=2):
        numeric_data = self.data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            raise ValueError("No numerical columns found in data")
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            pca.fit(numeric_data)
            reduced_data = pca.transform(numeric_data)
            return pd.DataFrame(reduced_data, columns=["Component " + str(i) for i in range(n_components)])
        else:
            raise ValueError("Unsupported dimensionality reduction method")

def main():
    st.set_page_config(page_title="EDA App", page_icon=":bar_chart:", layout="wide")
    st.title("Exploratory Data Analysis with Streamlit")

    # Upload the data file
    data_file = st.file_uploader("Upload data file", type=["csv", "xlsx"])

    if data_file is not None:
        # Load the data file into a Pandas DataFrame
        data = pd.read_csv(data_file)
        eda = EDA(data)

        # Display the summary statistics
        st.header("Summary Statistics")
        st.write(eda.summary_statistics())

        # Select columns to visualize
        columns = list(data.columns)
        column1 = st.selectbox("Select column 1", columns)
        column2 = st.selectbox("Select column 2", columns)

        # Display the correlation analysis
        st.header("Correlation Analysis")
        st.write(eda.correlation_analysis(column1, column2))

        # Select the type of visualization
        plot_type = st.selectbox("Select plot type", ["hist", "scatter", "box", "heatmap"])

        if plot_type == "scatter":
            # Select the X and Y columns for scatterplot
            x_column = st.selectbox("Select X column", columns)
            y_column = st.selectbox("Select Y column", columns)
        else:
            # Select the column for other types of visualizations
            column = st.selectbox("Select column", columns)

        # Display the visualization
        st.header("Data Visualization")
        plot_figure = None
        if plot_type == "scatter":
            plot_figure = eda.data_visualization(plot_type=plot_type, column=(x_column, y_column))
            plt.xlabel(x_column)
            plt.ylabel(y_column)
        else:
            plot_figure = eda
        if plot_figure is not None:
            st.pyplot(plot_figure)

        # Select the dimensionality reduction method
        st.header("Dimensionality Reduction")
        method = st.selectbox("Select dimensionality reduction method", ["pca"])
        n_components = st.slider("Select the number of components", min_value=1, max_value=len(data.columns))

        # Display the dimensionality reduction
        reduced_data = eda.dimensionality_reduction(method=method, n_components=n_components)
        st.write(reduced_data)

        # Plot the reduced data
        plot_figure = plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1])
        plt.xlabel("Component 0")
        plt.ylabel("Component 1")
        st.pyplot(plot_figure)

if __name__ == "__main__":
    main()
