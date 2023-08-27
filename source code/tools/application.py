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
from modeling import deal_with_nan
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import load
import folium
import matplotlib


def one_hot_encode_soiltype(df):
    """
    One-hot encode the 'soiltype' column in the given DataFrame.

    This function takes a DataFrame containing a 'soiltype' column, one-hot encodes this column,
    and then returns the DataFrame with the original 'soiltype' column replaced by the one-hot encoded columns.

    Parameters:
    -----------
    :param df: Input DataFrame containing the 'soiltype' column.
    :type df: pd.DataFrame
    :return: DataFrame with one-hot encoded 'soiltype' column.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> data = pd.DataFrame({'soiltype': ['A', 'B', 'A', 'C']})
        # >>> one_hot_encoded_data = one_hot_encode_soiltype(data)
    """
    # Generate dummy variables for the 'soiltype' column
    soiltype_dummies = pd.get_dummies(df['soiltype'], prefix='soiltype')
    # Drop the original 'soiltype' column
    df = df.drop(columns=['soiltype'])
    # Concatenate the original DataFrame and the dummy variables
    df = pd.concat([df, soiltype_dummies], axis=1)
    return df


def preprocess_application_data(data, unnecessary_cols_list, region_name_ls):
    """
    Preprocess application data for further analysis or modeling.

    This function carries out several preprocessing steps on the application data:
    1. One-hot encodes the 'soiltype' column.
    2. Adds a 'soiltype_Lf' column initialized with zeros.
    3. Drops unnecessary columns.
    4. Deals with missing values using the `deal_with_nan` function.

    Parameters:
    -----------
    :param data: Input application data DataFrame.
    :type data: pd.DataFrame
    :param unnecessary_cols_list: List of columns to be dropped from the DataFrame.
    :type unnecessary_cols_list: list of str
    :param region_name_ls: List of region names for handling missing values.
    :type region_name_ls: list of str
    :return: Preprocessed DataFrame.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> data = pd.DataFrame({'soiltype': ['A', 'B', 'A', 'C'], 'feature': [1, 2, 3, 4]})
        # >>> unnecessary_cols = ['feature']
        # >>> region_names = ['Region1', 'Region2']
        # >>> preprocessed_data = preprocess_application_data(data, unnecessary_cols, region_names)
    """

    # One-hot encode the 'soiltype' column
    data = one_hot_encode_soiltype(data)
    # Add a new column 'soiltype_Lf' initialized with zeros
    data['soiltype_Lf'] = 0
    # Drop unnecessary columns
    data.drop(columns=unnecessary_cols_list, inplace=True)
    # Deal with missing values using the `deal_with_nan` function
    data = deal_with_nan(data, region_name_ls, print_info=False)
    return data


def apply_model_and_evaluate(datasets, model_saved_path, unique_flood_id, plot=True, save_path=None):
    """
    Apply a previously saved model on datasets and evaluate its performance using confusion matrix.

    This function loads a previously saved model, applies it on the given datasets, and evaluates its
    performance using a confusion matrix. The results include accuracy, precision, recall, and F1 score,
    which are visualized using subplots for each dataset.


    Parameters:
    -----------
    :param datasets: List of datasets for which the model needs to be evaluated.
    :type datasets: list of pd.DataFrame
    :param model_saved_path: Path to the saved model without the iteration suffix.
    :type model_saved_path: str
    :param unique_flood_id: List of unique flood IDs corresponding to each dataset.
    :type unique_flood_id: list of str or int
    :param plot: Whether to plot the confusion matrices or not.
    :type plot: bool, default=True
    :param save_path: Path to save the confusion matrix plots, if required.
    :type save_path: str, default=None
    :return: List of tuples containing accuracy, precision, recall, and F1 score for each dataset.
    :rtype: list of tuple


    Example of usage:
    -----------------
        # >>> dataset1 = pd.DataFrame({'feature1': [1, 2, 3], 'label': [1, 0, 1]})
        # >>> dataset2 = pd.DataFrame({'feature1': [4, 5, 6], 'label': [0, 0, 1]})
        # >>> unique_ids = ['flood_1', 'flood_2']
        # >>> results = apply_model_and_evaluate([dataset1, dataset2], 'path_to_model', unique_ids)
    """

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    results = []

    for i, (data, flood_id) in enumerate(zip(datasets, unique_flood_id)):
        # Load the saved model
        loaded_model = load(f"{model_saved_path}_iter_{i + 1}.joblib")
        feature_order = loaded_model.feature_names_in_
        labels = data["label"]
        application_data = data.drop(columns=["label"])
        application_data_reordered = application_data[feature_order]
        predictions = loaded_model.predict(application_data_reordered)

        # Calculate confusion matrix and evaluation metrics
        cm = confusion_matrix(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        results.append((accuracy, precision, recall, f1))

        if plot:
            # Determine the subplot's position
            ax = axs[i // 3, i % 3]

            # Plot the confusion matrix on the subplot
            ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"Confusion Matrix of {flood_id}", fontsize=10)
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xlabel('Predicted label', fontsize=8)
            ax.set_ylabel('True label', fontsize=8)
            ax.tick_params(labelsize=7)

            # Add numeric labels within the subplot
            for j, k in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(k, j, cm[j, k], horizontalalignment="center", fontsize=8,
                        color="white" if cm[j, k] > cm.max() / 2 else "black")

            info_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}"
            ax.annotate(info_text, xy=(1.05, 0.5), xycoords="axes fraction", va="center", ha="left", fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle="round,pad=0.3"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return results


def get_coors_df(data, flood_id):
    """
    Extracts the longitude and latitude coordinates for a specific flood ID.

    This function filters a given dataset based on a specified flood ID and extracts
    the corresponding longitude (Lon) and latitude (Lat) coordinates.

    Parameters:
    -----------
    :param data: The dataset containing the flood data.
    :type data: pd.DataFrame
    :param flood_id: The flood ID for which coordinates are to be extracted.
    :type flood_id: str or int
    :return: DataFrame containing the longitude and latitude coordinates for the specified flood ID.
    :rtype: pd.DataFrame

    Example of usage:
    -----------------
        # >>> data = pd.DataFrame({'Flood_ID': ['flood_1', 'flood_1', 'flood_2'],
        #                           'Lon': [10, 20, 30], 'Lat': [15, 25, 35]})
        # >>> coors = get_coors_df(data, 'flood_1')
    """
    coors_df = data[data['Flood_ID'] == flood_id]
    return coors_df[['Lon', 'Lat']]


def get_actual_labels(data):
    """
    Extracts actual labels for each flood ID in the dataset.

    This function groups the data by flood ID and extracts the labels
    for each ID, storing them in a dictionary.

    Parameters:
    -----------
    :param data: The dataset containing the flood data.
    :type data: pd.DataFrame
    :return: Dictionary where keys are flood IDs and values are lists of labels for each flood ID.
    :rtype: dict

    Example of usage:
    -----------------
        # >>> data = pd.DataFrame({'Flood_ID': ['flood_1', 'flood_1', 'flood_2'],
        #                           'label': [1, 0, 1]})
        # >>> labels = get_actual_labels(data)
    """
    labels_dic = {}
    for id, df in data.groupby('Flood_ID'):
        labels_dic[id] = df["label"].tolist()
    return labels_dic


def plot_flood_maps_corrected(data_dict, coors_df1, coors_df2, predict_or_actual, save_path=None):
    """
    Plot flood maps based on given coordinates and labels.

    This function visualizes the flood maps for provided data. It uses folium to create maps and
    uses rectangles to indicate the regions of flooding. The color of the rectangles represents
    the intensity of flooding.

    Parameters:
    -----------
    :param data_dict: Dictionary containing flood IDs and their corresponding labels.
    :type data_dict: dict
    :param coors_df1: Coordinates DataFrame for the first region.
    :type coors_df1: pd.DataFrame
    :param coors_df2: Coordinates DataFrame for the second region.
    :type coors_df2: pd.DataFrame
    :param predict_or_actual: Specifies whether the data represents 'predicted' or 'actual' labels.
    :type predict_or_actual: str
    :param save_path: Optional. If provided, the generated maps will be saved at this path.
    :type save_path: str or None
    :return: List of folium Map objects.
    :rtype: list

    Example of usage:
    -----------------
        # >>> data = {'flood_1_subROI1': [1, 0, 1], 'flood_2_subROI2': [0, 1]}
        # >>> coors1 = pd.DataFrame({'Lon': [10, 20, 30], 'Lat': [15, 25, 35]})
        # >>> coors2 = pd.DataFrame({'Lon': [40, 50, 60], 'Lat': [45, 55, 65]})
        # >>> maps = plot_flood_maps_corrected(data, coors1, coors2, 'predicted')
    """

    # Function to get color based on label value
    def get_color(value):
        """Helper function to assign color based on the label value or probability."""
        if isinstance(value, int):
            return "red" if value == 1 else "grey"
        elif 0 <= value <= 1:
            cmap = plt.get_cmap('Reds')
            rgba = cmap(value - 0.01)
            return matplotlib.colors.rgb2hex(rgba)

    maps = []

    # Iterate through each flood_id and its associated labels
    for flood_id, labels in data_dict.items():
        # Determine the flood's region based on the flood_id
        if "subROI1" in flood_id:
            region_df = coors_df1
        else:
            region_df = coors_df2

        # Initialize the map for the region
        m = folium.Map(location=[region_df["Lat"].iloc[0], region_df["Lon"].iloc[0]], zoom_start=10)

        # Add rectangles for each coordinate in the region
        for (index, row), label in zip(region_df.iterrows(), labels):
            color = get_color(label)
            folium.Rectangle(
                bounds=[[row["Lat"] - 0.001125, row["Lon"] - 0.001125], [row["Lat"] + 0.001125, row["Lon"] + 0.001125]],
                color=color,
                fill=True,
                fill_color=color
            ).add_to(m)

        # Adding a legend to the map
        cmap = plt.get_cmap('Reds')
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 160px; height: 60px; z-index:9999; font-size:14px;">
        <div style="background-color: {}; width: 25%; height: 75%; float: left;"></div>
        <div style="background-color: {}; width: 25%; height: 75%; float: left;"></div>
        <div style="background-color: {}; width: 25%; height: 75%; float: left;"></div>
        <div style="background-color: {}; width: 25%; height: 75%; float: left;"></div>
        <br>&nbsp;&nbsp;Low &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;High
        </div>""".format(matplotlib.colors.rgb2hex(cmap(0.25)),
                         matplotlib.colors.rgb2hex(cmap(0.5)),
                         matplotlib.colors.rgb2hex(cmap(0.75)),
                         matplotlib.colors.rgb2hex(cmap(0.99)))
        m.get_root().html.add_child(folium.Element(legend_html))

        maps.append(m)

        # Save the map to a specified path if provided
        if save_path:
            m.save(f"{save_path}/{predict_or_actual}_{flood_id}.html")

    return maps