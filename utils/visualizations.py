import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords

class Visualization:
    """
    A class for visualizing data

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Visualization object
    - create_subplots: Create subplots for visualization
    - remove_unused_axes: Remove unused axes from subplots
    - plot_columns: Plot columns of the dataset using various plot types
    - numerical_univariate_analysis: Perform univariate analysis for numerical features
    - categorical_univariate_analysis: Perform univariate analysis for categorical features
    - num_features_vs_target: Plot numerical features against the target variable
    - cat_features_vs_target: Plot categorical features against the target variable using box plots
    - facegrid_hist_target: Plot scatterplots of numerical features against the target variable using FacetGrid
    - plot_scatter_numericals_target: Plot scatter plots for numerical features with the target variable as hue
    """
    def __init__(self, data):
        """
        Initialize the Visualization object

        Parameters:
        - data (DataFrame): The dataset to be visualized and analyzed
        """
        self.data = data

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple, optional): Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax (array of Axes): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def remove_unused_axes(self, fig, ax, num_plots):
        """
        Remove unused axes from the subplots grid

        Parameters:
        - fig (Figure): The figure object
        - ax (array of Axes): Array of axes objects
        - num_plots (int): Number of plots to be displayed
        """
        total_axes = len(ax)
        for j in range(num_plots, total_axes):
            fig.delaxes(ax[j])

    def plot_columns(self, cols, plot_func, ax, title_prefix="", target=None):
        """
        Plot columns from the dataset using the specified plotting function

        Parameters:
        - cols (list): List of column names to be plotted
        - plot_func (function): Plotting function (e.g., sns.boxplot, sns.histplot)
        - ax (array of Axes): Array of axes objects
        - title_prefix (str, optional): Prefix to be added to each plot title. Default is ""
        - target (str, optional): Target variable for plotting. Default is None
        """
        for i, col in enumerate(cols):
            if plot_func == sns.boxplot and target is not None:
                value_counts = self.data[col].value_counts()
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    plot_func(x=self.data[col], y=self.data[target], order=top_values.index, ax=ax[i])
                    ax[i].tick_params(axis="x", rotation=45)
                else:
                    plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
            elif plot_func == sns.histplot:
                plot_func(self.data[col], ax=ax[i], kde=True)
            elif plot_func == sns.scatterplot and target is not None:
                plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
            elif plot_func == sns.countplot:
                value_counts = self.data[col].value_counts()
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    count_ax = plot_func(x=self.data[col], order=top_values.index, ax=ax[i])
                    ax[i].tick_params(axis="x", rotation=90)
                else:
                    count_ax = plot_func(x=self.data[col], ax=ax[i])
                for container in count_ax.containers:
                    count_ax.bar_label(container)
            else:
                plot_func(x=self.data[col], ax=ax[i])
            ax[i].set_title(f"{title_prefix}{col.capitalize()}")

    def numerical_univariate_analysis(self, rows, columns):
        """
        Perform univariate analysis on numerical columns and plot distributions

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data.select_dtypes(include="number")
        self.plot_columns(cols, sns.histplot, ax, title_prefix="Distribution of ")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def categorical_univariate_analysis(self, rows, columns):
        """
        Performs univariate analysis on categorical columns and plot count plots

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data.select_dtypes("object")
        self.plot_columns(cols, sns.countplot, ax)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def num_features_vs_target(self, rows, columns, target):
        """
        Plots numerical features against the target variable

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - target (str): Name of the target variable
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(18, 12))
        cols = self.data.drop(columns=target).select_dtypes(include="number")
        self.plot_columns(cols, sns.scatterplot, ax, "Price x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def cat_features_vs_target(self, rows, columns, target, cols, figsize):
        """
        Plot categorical features against the target variable using box plots

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - target (str): Name of the target variable
        - cols (list): List of column names for plotting
        - figsize (tuple): Figure size
        """
        fig, ax = self.create_subplots(rows, columns, figsize=figsize)
        cols = self.data[cols]
        self.plot_columns(cols, sns.boxplot, ax, "Price x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def facegrid_hist_target(self, facecol, target):
        """
        Generates FacetGrid scatterplots for numerical columns based on target values

        Parameters:
        - facecol (str): Name of the column for the grid
        - target (str): Name of the target variable
        """
        for col in self.data.drop(columns=[target]).select_dtypes(include="number"):
            g = sns.FacetGrid(self.data, col=facecol)
            g.map(sns.scatterplot, col, target)
            plt.show()

    def plot_scatter_numericals_target(self, rows, columns, target, x):
        """
        Plot scatter plots for numerical features with the target variable as hue

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - target (str): Name of the target variable
        - x (str): Name of the feature variable for the x-axis
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(18, 12))
        cols = self.data.drop(columns=[target, x]).select_dtypes(include="number")
        for i, col in enumerate(cols):
            im = ax[i].scatter(y=self.data[col], x=self.data[x], c=self.data[target], cmap="tab20c", label="price", s=10)
            cbar = fig.colorbar(im, ax=ax[i], label="Price")
            ax[i].set_xlabel(x)
            ax[i].set_ylabel(col)
            ax[i].set_title(f"{x.capitalize()} x {col.capitalize()}")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

class TextPattern:
    """
    A class for analyzing patterns in text data

    Attributes:
    - ad_names (list): List of advertisement names

    Methods:
    - __init__: Initialize the TextPattern object
    - clean_text: Clean the text by removing non-alphanumeric characters
    - sort_words_by_frequency: Sort words by frequency
    - wordcloud: Generate and display a word cloud of advertisement names
    - ad_name_lengths: Plot the distribution of advertisement name lengths
    - word_frequencies: Plot the top 20 most frequent words in advertisement names
    - plot: Plot the data
    """
    def __init__(self, ad_names):
        """
        Initialize the TextPattern object

        Parameters:
        - ad_names (list): List of advertisement names
        """
        self.ad_names = ad_names
        self.stop_words = set(stopwords.words("english"))
    
    def clean_text(self, text):
        """
        Clean the text by removing non-alphanumeric characters

        Parameters:
        - text (str): Input text to be cleaned

        Returns:
        - str: Cleaned text
        """
        cleaned_text = re.sub(r'[^a-zA-Z\s]', "", text)
        return cleaned_text

    def sort_words_by_frequency(self, word_freq):
        """
        Sort words by frequency

        Parameters:
        - word_freq (Counter): Counter object containing word frequencies

        Returns:
        - tuple: Sorted words and their corresponding counts
        """
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_words = [word[0] for word in sorted_word_freq]
        sorted_counts = [word[1] for word in sorted_word_freq]
        return sorted_words, sorted_counts
    
    def wordcloud(self):
        """
        Generate and display a word cloud of advertisement names
        """
        cleaned_ads = [self.clean_text(ad_name) for ad_name in self.ad_names]
        cleaned_text = " ".join(cleaned_ads)
        self.plot(x=cleaned_text, title="Word Cloud of Ad Names")
    
    def ad_name_lengths(self):
        """
        Plot the distribution of advertisement name lengths
        """
        text_lengths = [len(ad_name) for ad_name in self.ad_names]
        self.plot(x=text_lengths, plot_func=sns.histplot, title="Distribution of Ad Name Lengths")
    
    def word_frequencies(self):
        """
        Plot the top 20 most frequent words in advertisement names
        """
        cleaned_ads = [self.clean_text(ad_name) for ad_name in self.ad_names]
        all_words = " ".join(cleaned_ads).split()
        word_freq = Counter([word for word in all_words if word.lower() not in self.stop_words])
        sorted_words, sorted_counts = self.sort_words_by_frequency(word_freq)
        self.plot(x=sorted_words[:20], y=sorted_counts[:20], plot_func=sns.barplot, title="Top 20 Most Frequent Words in Ad Names")

    def plot(self, x=None, y=None, plot_func=None, title=""):
        """
        Plot the data

        Parameters:
        - x (array-like): Data for the x-axis
        - y (array-like): Data for the y-axis
        - plot_func (function): Plotting function
        - title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        if plot_func == sns.histplot:
            plot_func(x=x, kde=True)
        elif plot_func == sns.barplot:
            bar_ax = plot_func(x=x, y=y)
            for container in bar_ax.containers:
                bar_ax.bar_label(container, fmt="%.2f")
            plt.xticks(rotation=45)
        else:
           wordcloud = WordCloud(width=800, height=400, background_color="white").generate(x)
           plt.imshow(wordcloud, interpolation="bilinear")
           plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()