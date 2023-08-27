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
import ee
import pandas as pd
import pickle
from tqdm.notebook import tqdm
import numpy as np


def download_daily_precip_dic(input_df, save_path=None, batch_size=1000, scale=5566):
    """
    Download daily precipitation data for given flood samples from CHIRPS dataset.

    This function fetches daily precipitation data for a given set of flood samples using the CHIRPS daily
    precipitation dataset available on Earth Engine. The data is downloaded in batches to prevent memory overflow.


    Parameters:
    -----------
    :param input_df: Input DataFrame containing flood samples with 'Lon', 'Lat', 'Event_Start_Date', and 'Event_End_Date'.
    :type input_df: pd.DataFrame
    :param save_path: Path to save the final precipitation dictionary, if desired.
    :type save_path: str, optional
    :param batch_size: Number of samples to process in each batch.
    :type batch_size: int, default 1000
    :param scale: The spatial resolution at which to sample the CHIRPS data, in meters.
    :type scale: float or int, default 5566
    :return: A dictionary containing the daily precipitation data for each flood sample.
    :rtype: dict


    Example of usage:
    -----------------
        # >>> samples_df = pd.DataFrame({
        # ...     'Lon': [0, 1],
        # ...     'Lat': [0, 1],
        # ...     'Event_Start_Date': ['2022-01-01', '2022-01-02'],
        # ...     'Event_End_Date': ['2022-01-10', '2022-01-12']
        # ... })
        # >>> precip_data = download_daily_precip_dic(samples_df)
    """

    # Copy input DataFrame to avoid modifying original data
    flood_samples = input_df.copy()
    # Initialize Earth Engine
    ee.Initialize()

    # Load CHIRPS daily precipitation data collection from Earth Engine
    chirps_daily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

    # Calculate the number of batches required
    num_batches = len(flood_samples) // batch_size + (len(flood_samples) % batch_size != 0)

    # Define function to get precipitation for a specific image and scale
    def get_precip(img, input_scale=scale):
        # Reduce the image over the region of the point
        precip_dict = img.reduceRegion(ee.Reducer.toList(), point, input_scale)
        # Get the precipitation value
        precip_value = precip_dict.get('precipitation')
        # Create a feature with the precipitation value
        precip_feature = ee.Feature(None, {'precipitation': precip_value})
        # Return the feature
        return precip_feature

    # Initialise dictionary to hold final precipitation values for all samples
    final_precip = {}

    # Create a progress bar to track the overall processing of all samples
    overall_progress = tqdm(total=len(flood_samples), desc='Daily Precipitation Overall Progress')

    # Process each batch of samples
    for batch_num in tqdm(range(num_batches), desc=f'Processing {num_batches} Batches daily precipitation'):
        # Get the start and end index for the current batch
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(flood_samples))

        # Get the current batch of samples
        batch_samples = flood_samples[start_index:end_index]

        # Create a dictionary to hold the precipitation values for the current batch
        batch_precip = {}

        # Process each sample in the current batch
        for index, row in batch_samples.iterrows():
            # Get the coordinate from the DataFrame
            coordinate = [row['Lon'], row['Lat']]
            # Create a point geometry
            point = ee.Geometry.Point(coordinate)
            # Get the start and end dates from the DataFrame
            start_date = pd.to_datetime(row['Event_Start_Date'])
            end_date = pd.to_datetime(row['Event_End_Date'])
            # Convert the dates to strings
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Filter the CHIRPS data
            chirps_filtered = chirps_daily.filterDate(start_date_str, end_date_str).filterBounds(point)

            # Apply the function to each Image in the ImageCollection
            precip_features = chirps_filtered.map(get_precip)

            # Convert the feature collection to a list and get the information
            precip_values = precip_features.aggregate_array('precipitation').getInfo()

            # Add the precipitation values to the dictionary
            batch_precip[f"{row['Flood_ID']}_{row['Lon']}_{row['Lat']}"] = precip_values

            # Update the overall progress bar
            overall_progress.update(1)

        # Merge the current batch dictionary with the final dictionary
        final_precip.update(batch_precip)

    # Close the overall progress bar
    overall_progress.close()

    # Save the final dictionary to a file if save_path is provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(final_precip, f)

    return final_precip


def calculate_rainfall_features(input_df, pickle_file_path, feature_calculators, column_names):
    """
    Calculate rainfall features for given flood samples based on stored precipitation data.

    This function computes specific rainfall features for a set of flood samples using pre-stored
    daily precipitation data. The features are calculated using the provided feature calculators
    and the results are added to new columns in the input DataFrame.


    Parameters:
    -----------
    :param input_df: Input DataFrame containing flood samples with 'Flood_ID', 'Lon', and 'Lat'.
    :type input_df: pd.DataFrame
    :param pickle_file_path: Path to the pickle file containing the daily precipitation data.
    :type pickle_file_path: str
    :param feature_calculators: List of functions that compute specific rainfall features.
    :type feature_calculators: list of callable
    :param column_names: List of column names corresponding to the computed features.
    :type column_names: list of str
    :return: DataFrame with the calculated rainfall features added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> sample_df = pd.DataFrame({
        # ...     'Flood_ID': ['A_1', 'B_2'],
        # ...     'Lon': [0, 1],
        # ...     'Lat': [0, 1],
        # ... })
        # >>> result_df = calculate_rainfall_features(sample_df, 'path_to_precip_data.pkl', [np.mean, np.sum], ['Avg_Rain', 'Total_Rain'])
    """

    # Copy input DataFrame to avoid modifying original data
    flood_samples = input_df.copy()

    # Load the daily precipitation data from the provided pickle file
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    # Define dictionaries to hold the computed features
    feature_dicts = [{} for _ in range(len(feature_calculators))]

    # Iterate through each key-value pair in the daily_rainfall dictionary
    for key, values in tqdm(data.items(), desc=f'Calculating <{column_names[0]}> features'):
        # Filter out None values
        values = [value for value in values if value is not None]

        # Calculate each feature using corresponding calculator
        for feature_calculator, feature_dict in zip(feature_calculators, feature_dicts):
            feature_dict[key] = feature_calculator(values)

    # Initialize lists to hold the feature values
    feature_lists = [[] for _ in range(len(feature_calculators))]

    # Process each row in the DataFrame
    for index, row in tqdm(flood_samples.iterrows(), total=len(flood_samples), desc=f'Processing <{column_names[0]}> column'):
        # Create the key from the Flood_ID, Lon, and Lat columns
        key = f"{row['Flood_ID']}_{row['Lon']}_{row['Lat']}"

        # Get the feature values from the dictionaries, if there is no corresponding key then return None
        for feature_dict, feature_list in zip(feature_dicts, feature_lists):
            feature_value = feature_dict.get(key, None)
            feature_list.append(feature_value)

    # Add the feature columns to the DataFrame
    for feature_list, column_name in zip(feature_lists, column_names):
        flood_samples[column_name] = feature_list
    return flood_samples


def flatten_list(rainfall_list):
    """
    Flatten a nested list if its elements are lists of single items.

    This function checks if the input list is nested with inner lists
    containing only one item. If it is, it flattens the list. Otherwise,
    it returns the original list.


    Parameters:
    -----------
    :param rainfall_list: List potentially containing nested lists.
    :type rainfall_list: list of any
    :return: A flattened version of the input list.
    :rtype: list of any


    Example of usage:
    -----------------
        # >>> flatten_list([[1], [2], [3]])
        # [1, 2, 3]
        # >>> flatten_list([1, 2, 3])
        # [1, 2, 3]
    """
    # Check if the first element is a list
    if isinstance(rainfall_list[0], list):
        # If it is, flatten the list by taking the first element of each inner list
        return [x[0] for x in rainfall_list]

    # If the list is not nested, return it as is
    return rainfall_list


def mean_calculator(values):
    """
    Calculate the mean of a list of values.

    This function checks if the input list is nested with inner lists
    containing only one item. If it is, it flattens the list using the
    flatten_list function. After ensuring the list is flat, it calculates
    and returns the mean of the values.


    Parameters:
    -----------
    :param values: List of numbers, possibly nested.
    :type values: list of float or int or list of list of float or int
    :return: Mean of the input values.
    :rtype: float


    Example of usage:
    -----------------
        # >>> mean_calculator([[1], [2], [3]])
        # 2.0
        # >>> mean_calculator([1, 2, 3])
        # 2.0
    """
    # Flatten the list if it is nested
    values = flatten_list(values)

    # Calculate and return the mean of the values
    return np.mean(values)


def median_calculator(values):
    """
    Calculate the median of a list of values.

    This function checks if the input list is nested with inner lists
    containing only one item. If it is, it flattens the list using the
    flatten_list function. After ensuring the list is flat, it calculates
    and returns the median of the values.


    Parameters:
    -----------
    :param values: List of numbers, possibly nested.
    :type values: list of float or int or list of list of float or int
    :return: Median of the input values.
    :rtype: float


    Example of usage:
    -----------------
        # >>> median_calculator([[1], [2], [3], [4]])
        # 2.5
        # >>> median_calculator([1, 2, 3, 4])
        # 2.5
    """

    # Flatten the list if it is nested
    values = flatten_list(values)
    # Calculate and return the median of the values
    return np.median(values)


def max_continuous_days_calculator(rainfall_list):
    """
    Calculate the maximum number of continuous rainy days from a list of rainfall values.

    This function checks the input list of rainfall values to find the longest
    sequence of days with positive rainfall (i.e., continuous rainy days).
    It returns the length of the longest sequence.


    Parameters:
    -----------
    :param rainfall_list: List of daily rainfall amounts.
    :type rainfall_list: list of float or int or list of list of float or int
    :return: Maximum number of continuous rainy days.
    :rtype: int


    Example of usage:
    -----------------
        # >>> max_continuous_days_calculator([[0], [2], [3], [0], [4], [5], [6]])
        # 3
        # >>> max_continuous_days_calculator([0, 2, 3, 0, 4, 5, 6])
        # 3
    """

    # Flatten the list if it is nested
    rainfall_list = flatten_list(rainfall_list)

    continuous_rainy_days = 0
    max_continuous_rainy_days = 0

    # Iterate through each day's rainfall
    for rainfall in rainfall_list:
        # Check if it rained that day
        if rainfall > 0:
            continuous_rainy_days += 1
        else:
            # Update the max count if the current streak is longer
            max_continuous_rainy_days = max(max_continuous_rainy_days, continuous_rainy_days)
            # Reset the current streak
            continuous_rainy_days = 0

    # return max_continuous_rainy_days
    # Handle cases where the longest streak is at the end of the list
    return max(max_continuous_rainy_days, continuous_rainy_days)


def max_continuous_rainfall_calculator(rainfall_list):
    """
    Calculate the maximum amount of continuous rainfall from a list of daily rainfall values.

    This function checks the input list of daily rainfall values to find the longest
    sequence of days with positive rainfall (i.e., continuous rainy days) and
    returns the total rainfall amount of that sequence.


    Parameters:
    -----------
    :param rainfall_list: List of daily rainfall amounts.
    :type rainfall_list: list of float or int or list of list of float or int
    :return: Maximum amount of continuous rainfall.
    :rtype: float


    Example of usage:
    -----------------
        # >>> max_continuous_rainfall_calculator([[0], [2], [3], [0], [4], [5], [6]])
        # 15
        # >>> max_continuous_rainfall_calculator([0, 2, 3, 0, 4, 5, 6])
        # 15
    """

    # Flatten the list if it is nested
    rainfall_list = flatten_list(rainfall_list)

    continuous_rainfall = 0
    max_continuous_rainfall = 0

    for rainfall in rainfall_list:
        if rainfall > 0:
            continuous_rainfall += rainfall
        else:
            max_continuous_rainfall = max(max_continuous_rainfall, continuous_rainfall)
            continuous_rainfall = 0

    # return max_continuous_rainfall

    # Handle cases where the highest accumulation is at the end of the list
    return max(max_continuous_rainfall, continuous_rainfall)


def max_single_day_rainfall_calculator(rainfall_list):
    """
    Calculate the maximum single-day rainfall amount from a list of daily rainfall values.

    This function checks the input list of daily rainfall values and returns the
    maximum rainfall amount observed in a single day.


    Parameters:
    -----------
    :param rainfall_list: List of daily rainfall amounts.
    :type rainfall_list: list of float or int or list of list of float or int
    :return: Maximum single-day rainfall amount.
    :rtype: float


    Example of usage:
    -----------------
        # >>> max_single_day_rainfall_calculator([[0], [2], [3], [0], [4], [5], [6]])
        # 6
        # >>> max_single_day_rainfall_calculator([0, 2, 3, 0, 4, 5, 6])
        # 6
    """

    # Flatten the list if it is nested
    rainfall_list = flatten_list(rainfall_list)
    # Return the maximum value from the rainfall list, default to 0 if the list is empty
    return max(rainfall_list, default=0)


def max_rainfall_increase_calculator(rainfall_list):
    """
    Calculate the maximum increase in rainfall from one day to the next.

    This function checks the input list of daily rainfall values and computes the
    difference between consecutive days. It then returns the maximum increase
    observed from one day to the next.


    Parameters:
    -----------
    :param rainfall_list: List of daily rainfall amounts.
    :type rainfall_list: list of float or int or list of list of float or int
    :return: Maximum increase in rainfall from one day to the next.
    :rtype: float


    Example of usage:
    -----------------
        # >>> max_rainfall_increase_calculator([[0], [2], [3], [0], [4], [5], [6]])
        # 4
        # >>> max_rainfall_increase_calculator([0, 2, 3, 0, 4, 5, 6])
        # 4
    """

    # Flatten the list if it is nested
    rainfall_list = flatten_list(rainfall_list)

    # Initialize the previous day's rainfall amount
    previous_rainfall = 0

    # Initialize the maximum rainfall increase
    max_rainfall_increase = 0

    # Iterate through the rainfall list
    for rainfall in rainfall_list:
        # Calculate the difference between the current day's rainfall and the previous day's
        difference = rainfall - previous_rainfall
        # Update the maximum rainfall increase if the current difference is greater
        max_rainfall_increase = max(max_rainfall_increase, difference)
        # Update the previous day's rainfall amount
        previous_rainfall = rainfall

    # Return the maximum rainfall increase
    return max_rainfall_increase


def add_mean_median_precip(input_df, daily_precip_pickle_file_path, save_path=None):
    """
    Add the mean and median daily rainfall columns to the input DataFrame.

    This function computes the mean and median daily rainfall values for each
    flood event in the provided DataFrame using daily rainfall data stored in
    a pickle file. After computation, it adds two new columns, 'Mean_Rainfall'
    and 'Median_Rainfall', to the input DataFrame.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param daily_precip_pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type daily_precip_pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with 'Mean_Rainfall' and 'Median_Rainfall' columns added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = add_mean_median_precip(df, 'daily_precip_data.pkl')
    """

    # Define feature calculators and column names
    feature_calculators = [mean_calculator, median_calculator]
    column_names = ['Mean_Rainfall', 'Median_Rainfall']

    # Call the general function with specific calculators and column names
    flood_samples = calculate_rainfall_features(input_df, daily_precip_pickle_file_path, feature_calculators, column_names)

    # Save the DataFrame to CSV if save_path is provided
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    # Return the updated DataFrame
    return flood_samples


def add_max_continuous_days_precip(input_df, pickle_file_path, save_path=None):
    """
    Add the maximum continuous rainy days column to the input DataFrame.

    This function computes the longest streak of continuous rainy days for each
    flood event in the provided DataFrame using daily rainfall data stored in
    a pickle file. After computation, it adds a new column, 'Max_Continuous_Rainy_Days',
    to the input DataFrame.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with 'Max_Continuous_Rainy_Days' column added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = add_max_continuous_days_precip(df, 'rainfall_data.pkl')
    """

    # Use the general function to calculate and add the Max_Continuous_Rainy_Days column
    flood_samples = calculate_rainfall_features(input_df, pickle_file_path, [max_continuous_days_calculator], ['Max_Continuous_Rainy_Days'])

    # If a save path is provided, save the updated DataFrame to a CSV file
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    # Return the updated DataFrame
    return flood_samples


def add_max_continuous_precip(input_df, pickle_file_path, save_path=None):
    """
    Add the maximum continuous rainfall column to the input DataFrame.

    This function computes the maximum continuous rainfall amount (i.e., the
    highest cumulative rainfall over consecutive rainy days) for each flood
    event in the provided DataFrame using daily rainfall data stored in a
    pickle file. After computation, it adds a new column, 'Max_Continuous_Rainfall',
    to the input DataFrame.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with 'Max_Continuous_Rainfall' column added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = add_max_continuous_precip(df, 'rainfall_data.pkl')
    """

    # Use the general function to calculate and add the Max_Continuous_Rainfall column
    flood_samples = calculate_rainfall_features(input_df, pickle_file_path, [max_continuous_rainfall_calculator], ['Max_Continuous_Rainfall'])
    # If a save path is provided, save the updated DataFrame to a CSV file
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    # Return the updated DataFrame
    return flood_samples


def add_max_single_day_precip(input_df, pickle_file_path, save_path=None):
    """
    Add the maximum single day rainfall column to the input DataFrame.

    This function computes the maximum rainfall amount recorded in a single day
    for each flood event in the provided DataFrame using daily rainfall data stored
    in a pickle file. After computation, it adds a new column, 'Max_Single_Day_Rainfall',
    to the input DataFrame.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with 'Max_Single_Day_Rainfall' column added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = add_max_single_day_precip(df, 'rainfall_data.pkl')
    """

    # Use the general function to calculate and add the Max_Single_Day_Rainfall column
    flood_samples = calculate_rainfall_features(input_df, pickle_file_path, [max_single_day_rainfall_calculator], ['Max_Single_Day_Rainfall'])
    # If a save path is provided, save the updated DataFrame to a CSV file
    if save_path:
        flood_samples.to_csv(save_path, index=False)
    # Return the updated DataFrame
    return flood_samples


def add_max_precip_increase(input_df, pickle_file_path, save_path=None):
    """
    Add the maximum rainfall increase column to the input DataFrame.

    This function computes the maximum increase in rainfall from one day to the next
    for each flood event in the provided DataFrame using daily rainfall data stored
    in a pickle file. After computation, it adds a new column, 'Max_Rainfall_Increase',
    to the input DataFrame.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with 'Max_Rainfall_Increase' column added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = add_max_precip_increase(df, 'rainfall_data.pkl')
    """

    # Use the general function to calculate and add the Max_Rainfall_Increase column
    flood_samples = calculate_rainfall_features(input_df, pickle_file_path, [max_rainfall_increase_calculator], ['Max_Rainfall_Increase'])
    # If a save path is provided, save the updated DataFrame to a CSV file
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    # Return the updated DataFrame
    return flood_samples


def combine_precip(input_df, daily_precip_pickle_file_path, save_path=None):
    """
    Combine various rainfall features into the input DataFrame.

    This function calculates and adds multiple rainfall features to the input DataFrame:
    - Mean_Rainfall
    - Median_Rainfall
    - Max_Continuous_Rainy_Days
    - Max_Continuous_Rainfall
    - Max_Single_Day_Rainfall
    - Max_Rainfall_Increase

    All calculations are based on daily rainfall data stored in a provided pickle file.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param daily_precip_pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type daily_precip_pickle_file_path: str
    :param save_path: Path where the updated DataFrame will be saved as a CSV file (optional).
    :type save_path: str or None, default None
    :return: DataFrame with multiple rainfall feature columns added.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({'Flood_ID': ['F1', 'F2'], 'Lon': [100, 101], 'Lat': [10, 11]})
        # >>> updated_df = combine_precip(df, 'rainfall_data.pkl')
    """

    # Add mean and median rainfall columns
    flood_samples = add_mean_median_precip(input_df, daily_precip_pickle_file_path)

    # Add the maximum continuous rainy days column
    flood_samples = add_max_continuous_days_precip(flood_samples, daily_precip_pickle_file_path)

    # Add the maximum continuous rainfall column
    flood_samples = add_max_continuous_precip(flood_samples, daily_precip_pickle_file_path)

    # Add the maximum single-day rainfall column
    flood_samples = add_max_single_day_precip(flood_samples, daily_precip_pickle_file_path)

    # Add the maximum rainfall increase column
    flood_samples = add_max_precip_increase(flood_samples, daily_precip_pickle_file_path)

    # If a save path is provided, save the updated DataFrame to a CSV file
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    # Return the updated DataFrame
    return flood_samples
