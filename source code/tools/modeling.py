"""
Author: Chaofan Wu
Student ID: 02285924
Email: cw522@ic.ac.uk
Project Name: Predicting flood risk in Ghana
Supervisors:
    Sesinam Dagadu(MEng)
    Yves Plancherel(PhD)
Company: SnooCODE
Date: 08/2023
"""
import pandas as pd
from tqdm.notebook import tqdm
from common_fucs import discard_duplicates
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
from joblib import dump


def load_data(path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    -----------
    :param path: Path to the CSV file.
    :type path: str
    :return: Data loaded from the CSV file into a pandas DataFrame.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> data = load_data("path/to/data.csv")
    """
    return pd.read_csv(path)


def check_missing_values(data):
    """
        Check and display columns with missing values from a pandas DataFrame.

        This function calculates the number of missing values in each column of the DataFrame and
        prints columns that have missing values along with their count.

        Parameters:
        -----------
        :param data: DataFrame for which missing values need to be checked.
        :type data: pd.DataFrame

        Example of usage:
        -----------------
            # >>> data1 = load_data("path/to/data.csv")
            # >>> check_missing_values(data1)
    """

    missing_values = data.isnull().sum()
    missing_values_cols = missing_values[missing_values > 0]
    print("Check how many missing values are in the following columns: ")
    print(missing_values_cols)
    print()
    print()


def deal_with_nan(input_df, region_name_ls, print_info=False, save_path=None):
    """
    Handle NaN values in the input DataFrame by filling them with regional mean values.

    The function processes the input DataFrame by regions, specified by the region_name_ls parameter.
    For each region, it identifies columns with missing values and fills these missing values
    with the mean of the non-missing values from the same column.

    Parameters:
    -----------
    :param input_df: Input DataFrame that contains missing values.
    :type input_df: pd.DataFrame
    :param region_name_ls: List of region names to be considered for filling NaN values.
    :type region_name_ls: list[str]
    :param print_info: Whether to print information about missing values before and after filling, default is False.
    :type print_info: bool
    :param save_path: Path where to save the resulting DataFrame as a CSV file, if provided.
    :type save_path: str or None, default None
    :return: DataFrame with NaN values filled with regional mean values.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> data2 = load_data("path/to/data.csv")
        # >>> regions = ['RegionA', 'RegionB']
        # >>> filled_data = deal_with_nan(data2, regions, print_info=True)
    """

    data = input_df.copy()

    filled_dataframes = []
    for region_name in region_name_ls:
        region_data = data[data['Flood_ID'].str.startswith(region_name)]

        # If print_info is True, display the columns with NaN values before filling
        if print_info:
            missing_values = region_data.isnull().sum()
            missing_values_cols = missing_values[missing_values > 0]
            print(f"Before filling the NaN in {region_name} has:")
            print(missing_values_cols)
            print()

        # Identify columns with missing values
        columns_with_nan = region_data.columns[region_data.isna().any()].tolist()

        # Calculate the mean values
        mean_values = region_data[columns_with_nan].mean()

        # Fill the missing values with the mean values
        region_data_filled = region_data.fillna(mean_values)

        # If print_info is True, display the columns with NaN values after filling
        if print_info:
            print(f"After filling the NaN in {region_name}:")
            missing_values_region = region_data_filled.isnull().sum()
            missing_values_cols = missing_values_region[missing_values_region > 0]
            print(missing_values_cols)
            print('----------------------------------------------------')

        filled_dataframes.append(region_data_filled)

    # Concatenate the filled dataframes
    full_data_filled = pd.concat(filled_dataframes)

    # If print_info is True, display the shape of the final filled DataFrame
    if print_info:
        print(f'The final data shape = {full_data_filled.shape}')

    # Save it to drive if save_path is provided
    if save_path:
        full_data_filled.to_csv(save_path, index=False)

    return full_data_filled


def get_datasets_for_Modeling(input_df, drop_flood_id=True, drop_duplicates=True, print_discard_duplicates_info=False):
    """
    Split the input DataFrame into multiple datasets based on the 'Flood_ID' and process each dataset.

    The function processes the input DataFrame by splitting it into subsets based on the 'Flood_ID'.
    It offers options to drop the 'Flood_ID' column and to discard duplicates within each subset.


    Parameters:
    -----------
    :param input_df: Input DataFrame to be split and processed.
    :type input_df: pd.DataFrame
    :param drop_flood_id: Whether to drop the 'Flood_ID' column from each subset, default is True.
    :type drop_flood_id: bool
    :param drop_duplicates: Whether to discard duplicates within each subset, default is True.
    :type drop_duplicates: bool
    :param print_discard_duplicates_info: Whether to print information during the duplicates discarding process, default is False.
    :type print_discard_duplicates_info: bool
    :return: A list of datasets (DataFrames) processed as per the function parameters.
    :rtype: list[pd.DataFrame]


    Example of usage:
    -----------------
        # >>> data = load_data("path/to/data.csv")
        # >>> datasets1 = get_datasets_for_Modeling(data, drop_flood_id=True, drop_duplicates=True, print_discard_duplicates_info=True)
    """

    data_df = input_df.copy()

    # Split the dataset into subsets based on the 'Flood_ID' column
    datasets = [subset for _, subset in tqdm(data_df.groupby('Flood_ID'), desc='Splitting dataset')]

    # Check duplicates and drop 'Flood_ID' if needed
    for i in tqdm(range(len(datasets)), desc='Processing subsets'):
        # Drop the 'Flood_ID' column if the flag is set
        if drop_flood_id:
            datasets[i].drop(columns=['Flood_ID'], inplace=True)
        # Check for duplicates and remove them if the flag is set
        if drop_duplicates:
            if print_discard_duplicates_info:
                print(f"Processing duplicates for subset {i+1}:")

            # Call the function to discard duplicates
            datasets[i] = discard_duplicates(datasets[i], print_info=print_discard_duplicates_info)

            if print_discard_duplicates_info:
                print("-----------------------------------------------")

    return datasets


def preprocess_datasets(datasets, i, random_state_seed=42):
    """
    Preprocesses datasets for modeling, including splitting, sampling, and one-hot encoding.

    The function separates the datasets into train, validation, and test datasets. It then
    samples the training datasets, one-hot encodes categorical variables, and ensures that
    all datasets have matching columns.


    Parameters:
    -----------
    :param datasets: List of datasets (DataFrames) to be processed.
    :type datasets: list[pd.DataFrame]
    :param i: Index of the dataset that should be used as the test dataset.
    :type i: int
    :param random_state_seed: Random seed for reproducibility in sampling.
    :type random_state_seed: int
    :return: Features and labels for training, validation, and test datasets.
    :rtype: tuple(pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series)


    Example of usage:
    -----------------
        # >>> datasets2 = [df1, df2, df3, df4]
        # >>> X_train1, y_train1, X_val1, y_val1, X_test1, y_test1 = preprocess_datasets(datasets2, 2)
    """

    # Create the test dataset
    X_test = datasets[i].drop(columns=['label'])
    y_test = datasets[i]['label']

    # Create the train dataset
    train_datasets = datasets[:i] + datasets[i + 1:]

    # Sampling datasets based on a fixed percentage
    sampled_train_datasets = [dataset.sample(frac=0.8, random_state=random_state_seed) for dataset in train_datasets]
    sampled_val_datasets = [dataset.drop(sampled_train_datasets[j].index) for j, dataset in enumerate(train_datasets)]

    # Combining the sampled datasets
    train_data = pd.concat(sampled_train_datasets)
    val_data = pd.concat(sampled_val_datasets)

    # Separate features and labels for training data
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    # Separate features and labels for validation data
    X_val = val_data.drop(columns=['label'])
    y_val = val_data['label']

    # One-Hot Encoding for categorical variables
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_test = pd.get_dummies(X_test)

    # Ensuring the columns match across datasets
    combined = pd.concat([X_train, X_val, X_test]).fillna(0)
    X_train = combined.iloc[:X_train.shape[0]].copy()
    X_val = combined.iloc[X_train.shape[0]:X_train.shape[0] + X_val.shape[0]].copy()
    X_test = combined.iloc[X_train.shape[0] + X_val.shape[0]:].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


def set_seeds(seed_value):
    """
    Set seeds for numpy, tensorflow, and python's random module to ensure reproducibility.

    By setting the seeds for these libraries, you ensure that the random processes in your
    code produce the same results every time they are run. This is essential for reproducibility
    in scientific experiments, model training, and other processes that rely on randomness.

    Parameters:
    -----------
    :param seed_value: The seed value to be set for the random processes.
    :type seed_value: int

    Example of usage:
    -----------------
        # >>> set_seeds(42)
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)


def train_deep_learning_model(X_train, y_train, X_val, y_val, input_dim,
                              epochs=60,
                              seed_value=42,
                              verbose=0,
                              save_model_path=None,
                              iteration=None):
    """
        Train a deep learning model for binary classification.

        This function constructs and trains a simple feedforward neural network for binary classification.
        The model is trained using the provided training and validation data and can be saved to disk if desired.


        Parameters:
        -----------
        :param X_train: Training features.
        :type X_train: pd.DataFrame or np.array
        :param y_train: Training labels.
        :type y_train: pd.Series or np.array
        :param X_val: Validation features.
        :type X_val: pd.DataFrame or np.array
        :param y_val: Validation labels.
        :type y_val: pd.Series or np.array
        :param input_dim: Number of input features.
        :type input_dim: int
        :param epochs: Number of training epochs. Default is 60.
        :type epochs: int
        :param seed_value: Seed value for random processes to ensure reproducibility. Default is 42.
        :type seed_value: int
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 0.
        :type verbose: int
        :param save_model_path: Path to save the trained model. If provided, the model will be saved to this path. Default is None.
        :type save_model_path: str or None
        :param iteration: Current iteration (useful for saving models during cross-validation). Default is None.
        :type iteration: int or None
        :return: Trained deep learning model.
        :rtype: keras.models.Sequential


        Example of usage:
        -----------------
            # >>> model1 = train_deep_learning_model(X_train, y_train, X_val, y_val, input_dim=100, epochs=50, save_model_path='./model')
    """

    # Ensure reproducibility by setting seeds
    set_seeds(seed_value)
    # Build the neural network model
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with binary cross-entropy loss
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model using the provided data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=verbose)

    # If a save path is provided, save the model to disk
    if save_model_path and iteration is not None:
        model.save(f"{save_model_path}_iter_{iteration}.h5")

    return model


def train_other_model(model, X_train, y_train, save_model_path=None, iteration=None):
    """
    Train a machine learning model and optionally save it to disk.

    This function is designed to train non-deep learning models (e.g., RandomForest, SVM) using the provided
    training data. After training, the model can be saved to disk using the joblib library if desired.


    Parameters:
    -----------
    :param model: An initialized machine learning model (e.g., from scikit-learn).
    :type model: sklearn.base.BaseEstimator
    :param X_train: Training features.
    :type X_train: pd.DataFrame or np.array
    :param y_train: Training labels.
    :type y_train: pd.Series or np.array
    :param save_model_path: Path to save the trained model. If provided, the model will be saved to this path. Default is None.
    :type save_model_path: str or None
    :param iteration: Current iteration (useful for saving models during cross-validation). Default is None.
    :type iteration: int or None
    :return: Trained machine learning model.
    :rtype: sklearn.base.BaseEstimator


    Example of usage:
    -----------------
        # >>> from sklearn.ensemble import RandomForestClassifier
        # >>> model12 = RandomForestClassifier()
        # >>> trained_model = train_other_model(model12, X_train, y_train, save_model_path='./model_rf')
    """

    # Train the provided model using the given training data
    model.fit(X_train, y_train)

    # If a save path is provided, save the model to disk using joblib
    if save_model_path and iteration is not None:
        dump(model, f"{save_model_path}_iter_{iteration}.joblib")

    return model


def evaluate_predictions(y_val, y_val_pred, y_test, y_test_pred, model_number):
    """
    Evaluate the performance of a model's predictions using various metrics.

    This function computes accuracy, F1 score, precision, and recall for the model's predictions on both
    validation and test datasets. It returns these metrics in a dictionary format.


    Parameters:
    -----------
    :param y_val: True labels for the validation set.
    :type y_val: pd.Series or np.array
    :param y_val_pred: Predicted labels for the validation set.
    :type y_val_pred: pd.Series or np.array
    :param y_test: True labels for the test set.
    :type y_test: pd.Series or np.array
    :param y_test_pred: Predicted labels for the test set.
    :type y_test_pred: pd.Series or np.array
    :param model_number: Identifier for the model being evaluated.
    :type model_number: int or str
    :return: Dictionary containing the computed evaluation metrics for the model.
    :rtype: dict


    Example of usage:
    -----------------
        # >>> y_val_true = [1, 0, 1, 1, 0]
        # >>> y_val_pred12 = [1, 0, 0, 1, 1]
        # >>> y_test_true = [0, 1, 0, 1, 1]
        # >>> y_test_pred12 = [0, 1, 0, 0, 1]
        # >>> results = evaluate_predictions(y_val_true, y_val_pred12, y_test_true, y_test_pred12, model_number=1)
    """

    # Compute the evaluation metrics for validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=1)
    val_precision = precision_score(y_val, y_val_pred, zero_division=1)
    val_recall = recall_score(y_val, y_val_pred, zero_division=1)

    # Compute the evaluation metrics for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=1)
    test_precision = precision_score(y_test, y_test_pred, zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, zero_division=1)

    # Return the metrics in dictionary format
    return {
        'Model': model_number,
        'Validation Accuracy': val_accuracy,
        'Validation F1': val_f1,
        'Validation Precision': val_precision,
        'Validation Recall': val_recall,
        'Test Accuracy': test_accuracy,
        'Test F1': test_f1,
        'Test Precision': test_precision,
        'Test Recall': test_recall
    }


def train_and_evaluate_model(datasets, model, model_name,
                             deep_learning_epoch=60, deep_learning_verbose=0,
                             save_model_path=None,
                             save_metrics_path=None,
                             random_state_seed=42, plot_metrics=False):
    """
    Train and evaluate the performance of a given model on multiple datasets.

    This function trains the model on multiple datasets, evaluates its performance on both validation
    and test datasets, and returns these metrics in a DataFrame format.


    Parameters:
    -----------
    :param datasets: List of datasets to train and evaluate the model on.
    :type datasets: list of pd.DataFrame
    :param model: The machine learning or deep learning model to be trained.
    :type model: Model object
    :param model_name: Name of the model ("Neural Net" for deep learning models, or other names for traditional ML models).
    :type model_name: str
    :param deep_learning_epoch: Number of epochs for training the deep learning model.
    :type deep_learning_epoch: int, optional, default is 60
    :param deep_learning_verbose: Verbosity mode for training the deep learning model.
    :type deep_learning_verbose: int, optional, default is 0
    :param save_model_path: Path to save the trained model.
    :type save_model_path: str, optional, default is None
    :param save_metrics_path: Path to save the model's performance metrics.
    :type save_metrics_path: str, optional, default is None
    :param random_state_seed: Random seed for reproducibility.
    :type random_state_seed: int, optional, default is 42
    :param plot_metrics: Whether to plot the evaluation metrics.
    :type plot_metrics: bool, optional, default is False


    Returns:
    --------
    :return: DataFrame containing the computed evaluation metrics for the model on each dataset.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> datasets1 = [df11, df21, df31]
        # >>> model1 = Sequential()
        # >>> results = train_and_evaluate_model(datasets1, model1, "Neural Net")
    """

    metrics = []
    for i in range(len(datasets)):
        # Preprocess the datasets
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_datasets(datasets, i, random_state_seed)
        if model_name == "Neural Net":
            # Scaling for Deep Learning
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Train the deep learning model
            trained_model = train_deep_learning_model(X_train, y_train, X_val, y_val, X_train.shape[1],
                                                      epochs=deep_learning_epoch,
                                                      seed_value=random_state_seed,
                                                      verbose=deep_learning_verbose,
                                                      save_model_path=save_model_path,
                                                      iteration=i + 1)
            # Predict the labels
            y_val_pred = (trained_model.predict(X_val) > 0.5).astype("int32")
            y_test_pred = (trained_model.predict(X_test) > 0.5).astype("int32")
        else:
            # Train other types of models
            trained_model = train_other_model(model, X_train, y_train, save_model_path=save_model_path, iteration=i + 1)
            y_val_pred = trained_model.predict(X_val)
            y_test_pred = trained_model.predict(X_test)

        # Evaluate the model's performance and append to the metrics list
        metrics.append(evaluate_predictions(y_val, y_val_pred, y_test, y_test_pred, i + 1))

    # Convert the metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Calculate mean performance for the selected columns
    mean_performance = metrics_df.drop(columns=['Model']).mean()
    mean_performance['Model'] = "Overall Performance"

    # Append the mean performance to the dataframe using concat
    mean_performance_df = pd.DataFrame(mean_performance).T
    metrics_df = pd.concat([metrics_df, mean_performance_df], ignore_index=True)

    # Convert 'Model' column to string type for compatibility with plotting
    metrics_df['Model'] = metrics_df['Model'].astype(str)

    # Plot the metrics if specified
    if plot_metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_df['Model'], metrics_df['Test F1'], label='Test F1')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} - Test Accuracy')
        plt.legend()
        plt.show()

    # Save the metrics if a save path is provided
    if save_metrics_path:
        metrics_df.to_csv(save_metrics_path, index=False)
        print(f"your {model_name} performance metric file(.csv) has been saved to {save_metrics_path} ")

    return metrics_df


def plot_model_performance(models_dict, save_path=None):
    """
    Plot the performance metrics comparison for different models.

    This function visualizes the overall performance metrics of different models on a bar chart.
    It allows for a clear comparison across models in terms of accuracy, precision, recall, and F1 score
    for both validation and test datasets.


    Parameters:
    -----------
    :param models_dict: Dictionary where the key is the model's name and the value is its performance metrics DataFrame.
    :type models_dict: dict of {str: pd.DataFrame}
    :param save_path: Optional path to save the generated plot.
    :type save_path: str, optional, default is None


    Example of usage:
    -----------------
        # >>> models_performance = {
        # ...     "Neural Net": neural_net_df,
        # ...     "Random Forest": random_forest_df
        # ... }
        # >>> plot_model_performance(models_performance, save_path="performance_comparison.png")
    """

    # Names of the metrics for plotting
    metrics_names = [
        "Overall Val Acc", "Overall Val Precision", "Overall Val Recall", "Overall Val F1",
        "Overall Test Acc", "Overall Test Precision", "Overall Test Recall", "Overall Test F1"
    ]

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define bar width and initial positions
    bar_width = 0.12
    bar_positions = range(len(metrics_names))

    # Plot bars for each model's performance metrics
    for idx, (model_name, df) in enumerate(models_dict.items()):
        # Extract overall performance from the dataframe
        overall_performance = df.loc[df['Model'] == 'Overall Performance']
        # List of performance metrics values to be plotted
        performance_values = [
            overall_performance['Validation Accuracy'].values[0],
            overall_performance['Validation Precision'].values[0],
            overall_performance['Validation Recall'].values[0],
            overall_performance['Validation F1'].values[0],
            overall_performance['Test Accuracy'].values[0],
            overall_performance['Test Precision'].values[0],
            overall_performance['Test Recall'].values[0],
            overall_performance['Test F1'].values[0]
        ]
        # Adjust positions for each model's bars
        positions = [pos + idx * bar_width for pos in bar_positions]
        # Plot the bars
        ax.bar(positions, performance_values, width=bar_width, color=plt.cm.tab10(idx), label=model_name)

    # Adjust the x-axis ticks positions and labels
    tick_positions = [pos + bar_width * (len(models_dict) / 2) - bar_width / 2 for pos in bar_positions]
    plt.xticks(tick_positions, metrics_names, rotation=45, ha='right')

    # Set y-axis label
    plt.ylabel('Performance')

    # Add legend to the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(models_dict))

    # Add a title to the plot
    plt.title('Model Performance Comparison', fontsize=18)

    # Save the plot to a file if a save_path is provided
    if save_path:
        plt.savefig(save_path, format='png' if save_path.endswith('.png') else 'jpg', bbox_inches='tight')

    # Display the plot
    plt.tight_layout()
    plt.show()