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
import ee
from tqdm.notebook import tqdm
import numpy as np


def initialize_ee_and_get_flood_data():
    """
    Initialize Earth Engine and retrieve the global flood dataset.

    This function initializes the Earth Engine and then fetches the global flood dataset
    from the "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1" collection.


    Parameters:
    -----------
    :return: Global flood dataset from the Earth Engine collection.
    :rtype: ee.ImageCollection


    Example of usage:
    -----------------
        # >>> flood_data = initialize_ee_and_get_flood_data()
    """

    ee.Initialize()
    return ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")


def calculate_flooded_area_proportion(flood_event, ROI_boundary, scale):
    """
    Calculate the proportion of the flooded area for a given flood event within a specified ROI.

    This function determines the proportion of the area that got flooded during a specific flood event
    within the given Region Of Interest (ROI). It takes into account only those areas that are
    flooded and are not permanent water bodies.


    Parameters:
    -----------
    :param flood_event: The flood event data to be analyzed.
    :type flood_event: ee.Image
    :param ROI_boundary: The boundary of the Region Of Interest (ROI) where the flood proportion needs to be calculated.
    :type ROI_boundary: ee.Geometry or ee.Feature or ee.FeatureCollection
    :param scale: The spatial resolution of the flood data in meters.
    :type scale: float or int
    :return: Proportion of the area that got flooded within the given ROI.
    :rtype: float


    Example of usage:
    -----------------
        # >>> event_id = 2320
        # >>> global_flood_data = get_flood_sampling_data.initialize_ee_and_get_flood_data()
        # >>> flood_event = global_flood_data.filter(ee.Filter.eq('id', event_id)).first()
        # >>> ROI = ee.Geometry.Rectangle([-0.87, 8.05, -0.37, 8.55])
        # >>> proportion = calculate_flooded_area_proportion(flood_data, ROI, 250)
    """

    # Create an image representation of the flooded area, excluding permanent water bodies.
    flooded_and_not_perm_water = flood_event.expression(
        "(flooded == 1) && (jrc_perm_water == 0)",
        {'flooded': flood_event.select('flooded'), 'jrc_perm_water': flood_event.select('jrc_perm_water')}
    ).eq(1).toByte().rename('class')

    # Count the number of pixels that represent flooded areas within the ROI.
    flood_pixel_count = flooded_and_not_perm_water.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=ROI_boundary, scale=scale).get('class').getInfo()

    # Count the total number of pixels within the ROI.
    total_pixel_count = flooded_and_not_perm_water.unmask().reduceRegion(
        reducer=ee.Reducer.count(), geometry=ROI_boundary, scale=scale).get('class').getInfo()

    # Calculate and return the proportion of flooded pixels to the total pixels.
    return flood_pixel_count / total_pixel_count


def stratified_sampling(flood_event, ROI_boundary, numPoints, random_seed, scale):
    """
    Perform stratified sampling on a flood event within a specified Region Of Interest (ROI).

    This function conducts stratified sampling on areas of a flood event that are flooded
    but are not permanent water bodies. Stratified sampling ensures that the sample represents
    the underlying stratification of the data.


    Parameters:
    -----------
    :param flood_event: The flood event data to be sampled.
    :type flood_event: ee.Image
    :param ROI_boundary: The boundary of the Region Of Interest (ROI) where sampling should be conducted.
    :type ROI_boundary: ee.Geometry or ee.Feature or ee.FeatureCollection
    :param numPoints: Number of sampling points to be collected.
    :type numPoints: int
    :param random_seed: Seed for the random sampling process to ensure reproducibility.
    :type random_seed: int
    :param scale: The spatial resolution of the flood data in meters.
    :type scale: float or int
    :return: List of sampled features with their properties and geometries.
    :rtype: list

    """

    # Create an image representation of the flooded area, excluding permanent water bodies.
    flooded_and_not_perm_water = flood_event.expression(
        "(flooded == 1) && (jrc_perm_water == 0)",
        {'flooded': flood_event.select('flooded'), 'jrc_perm_water': flood_event.select('jrc_perm_water')}
    )
    flooded_and_not_perm_water_img = flooded_and_not_perm_water.eq(1).toByte().rename('class')

    # Execute stratified sampling on the created image.
    samples = flooded_and_not_perm_water_img.stratifiedSample(
        numPoints=numPoints,
        classBand='class',
        region=ROI_boundary,
        scale=scale,
        seed=random_seed,
        dropNulls=True,
        geometries=True
    )

    # Extract and return the sampled features.
    return samples.getInfo()['features']


def auto_get_flood_data_dic_from_a_ROI(flood_ids, ROI_boundary, total_num_samples, region_name, scale=30,
                                   check_each_sample_amount=False, print_info=False):
    """
    Automatically retrieves flood data for a specified Region of Interest (ROI).

    This function performs stratified sampling on flood data within a given ROI. It provides
    coordinates for both flooded and non-flooded areas based on the given flood event IDs.


    Parameters:
    -----------
    :param flood_ids: List of flood event IDs for which data is to be fetched.
    :type flood_ids: list
    :param ROI_boundary: The boundary of the region of interest.
    :type ROI_boundary: ee.Geometry.Polygon or ee.Geometry.Rectangle
    :param total_num_samples: Total number of samples to be collected across all events.
    :type total_num_samples: int
    :param region_name: Name of the region being processed.
    :type region_name: str
    :param scale: The resolution at which to retrieve data.
    :type scale: int, optional
    :param check_each_sample_amount: Flag to check the number of samples for each event.
    :type check_each_sample_amount: bool, optional
    :param print_info: Flag to print progress and diagnostic information.
    :type print_info: bool, optional
    :return: Two dictionaries - first one contains coordinates of flooded areas for each event ID
             and the second one contains coordinates of non-flooded areas for each event ID.
    :rtype: tuple of two dictionaries

    """

    global_flood_data = initialize_ee_and_get_flood_data()
    flooded_coordinates_dict = {}
    none_flooded_coordinates_dict = {}
    total_flooded_area_proportion = 0

    if print_info:
        print("Start Sampling......")
        print('---------------------------------------------------------------')

    # Calculate total flooded area proportion
    for event_id in tqdm(flood_ids, desc="Calculating flooded area proportion", disable=not print_info):
        flood_event = global_flood_data.filter(ee.Filter.eq('id', event_id)).first()
        flooded_area_proportion = calculate_flooded_area_proportion(flood_event, ROI_boundary, scale=scale)
        total_flooded_area_proportion += flooded_area_proportion

    # Perform stratified sampling
    for event_id in tqdm(flood_ids, desc="Performing stratified sampling", disable=not print_info):
        flood_event = global_flood_data.filter(ee.Filter.eq('id', event_id)).first()
        flooded_area_proportion = calculate_flooded_area_proportion(flood_event, ROI_boundary, scale=scale)
        numPoints = int(total_num_samples * (flooded_area_proportion / total_flooded_area_proportion) / 2)

        # Get the features
        features = stratified_sampling(flood_event, ROI_boundary, numPoints, random_seed=event_id, scale=scale)
        flooded_coordinates_dict[event_id] = [f['geometry']['coordinates'] for f in features if
                                              f['properties']['class'] == 1]
        none_flooded_coordinates_dict[event_id] = [f['geometry']['coordinates'] for f in features if
                                                   f['properties']['class'] == 0]

    if print_info:
        print('---------------------------------------------------------------')
        print("Finish Sampling")
        print()
        print()

    if check_each_sample_amount:
        print("Check the amount of each flooded sampling:")
        flooded_count = sum(len(v) for v in flooded_coordinates_dict.values())
        none_flooded_count = sum(len(v) for v in none_flooded_coordinates_dict.values())
        print(f'Total flooded samples in {region_name} = {flooded_count}')
        print(f'Total non-flooded samples in {region_name} = {none_flooded_count}')
        print(f'Total samples in {region_name} = {flooded_count + none_flooded_count}')

    return flooded_coordinates_dict, none_flooded_coordinates_dict


def coordinates_dict_to_df(coordinates_dict, flooded, prefix):
    """
    Convert a dictionary of coordinates into a DataFrame.

    This function processes a dictionary containing flood event IDs and their associated
    coordinates. It converts this dictionary into a DataFrame format, making it suitable for
    further analysis or storage. The function is flexible and can handle both modeling and
    application data based on the length of the coordinates provided.


    Parameters:
    -----------
    :param coordinates_dict: Dictionary containing flood event IDs as keys and lists of
                             coordinates as values.
    :type coordinates_dict: dict
    :param flooded: Binary value indicating whether the area corresponding to the coordinates
                    was flooded (used for modeling data).
    :type flooded: int
    :param prefix: Prefix to be added to the flood event ID to differentiate between different
                   types or sources of data.
    :type prefix: str
    :return: DataFrame containing the flood event IDs and their associated coordinates.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> coordinates_dict = {'event1': [(1.0, 1.0), (2.0, 2.0)], 'event2': [(3.0, 3.0, 1), (4.0, 4.0, 0)]}
        # >>> flooded = 1
        # >>> prefix = 'test'
        # >>> output_df = get_flood_sampling_data.coordinates_dict_to_df(coordinates_dict, flooded, prefix)
    """

    # List to store rows of the DataFrame.
    df_list = []
    # Iterate over each event ID and its associated coordinates.
    for event_id, coordinates_list in coordinates_dict.items():
        for coordinates in coordinates_list:
            # Dictionary to represent a row in the DataFrame.
            row_dict = {}
            # If the length of coordinates is 2, this is modeling data.
            if len(coordinates) == 2:
                row_dict = {'Flood_ID': prefix + "_" + str(event_id), 'Lon': coordinates[0], 'Lat': coordinates[1],
                            'Flooded': flooded}
            # If the length of coordinates is 3, this is application data.
            if len(coordinates) == 3:
                row_dict = {'Flood_ID': prefix + "_" + str(event_id), 'Lon': coordinates[0], 'Lat': coordinates[1],
                            'Flooded': coordinates[2]}
            # Append the row dictionary to the list.
            df_list.append(row_dict)

    # Convert the list of dictionaries into a DataFrame and return.
    return pd.DataFrame(df_list)


def auto_get_flood_data_df_from_a_list_ROIs(flood_ids, ROI_features_ls_3d, scale=30,
                                            check_each_sample_amount=False, print_info=True):
    """
    Automatically retrieve flood data for a list of Regions Of Interest (ROIs) and compile them into a DataFrame.

    This function processes a list of ROIs and for each ROI, fetches the flood data based on the provided
    flood event IDs. It then compiles this data into a DataFrame for further analysis or storage.


    Parameters:
    -----------
    :param flood_ids: List of flood event IDs to be fetched.
    :type flood_ids: list
    :param ROI_features_ls_3d: List of 3D features containing information about the ROI's boundary, total
                               number of samples, and region name.
    :type ROI_features_ls_3d: list
    :param scale: The spatial resolution of the flood data in meters.
    :type scale: float or int, default 30
    :param check_each_sample_amount: Flag to determine whether to check the amount of each sample.
    :type check_each_sample_amount: bool, default False
    :param print_info: Flag to determine whether to print information about the flood data fetching process.
    :type print_info: bool, default True
    :return: DataFrame containing flood data for all specified ROIs.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> flood_ids_list = ['event1', 'event2']
        # >>> ROIs = [[[0, 0, 5, 5], [100, 'Region1']], [[5, 5, 10, 10], [200, 'Region2']]]
        # >>> df = auto_get_flood_data_df_from_a_list_ROIs(flood_ids_list, ROIs)
    """

    # List to store DataFrames containing flood data for each ROI.
    all_flood_coors_df = []

    # Process each set of ROI features.
    for i in ROI_features_ls_3d:
        # Extract the boundary of the ROI.
        ROI_boundary = ee.Geometry.Rectangle(i[0])
        # Extract the total number of samples and the region name.
        total_num_samples = i[1][0]
        region_name = i[2][0]
        # Fetch flood data for the ROI using the specified function: auto_get_flood_data_dic_from_a_ROI.
        flooded_coors_dic, none_flooded_coors_dic = auto_get_flood_data_dic_from_a_ROI(flood_ids,
                                                ROI_boundary,
                                                total_num_samples,
                                                region_name,
                                                scale=scale,
                                                check_each_sample_amount=check_each_sample_amount,
                                                print_info=print_info)

        # Convert the coordinates dictionary to a DataFrame.
        flooded_coors_df = coordinates_dict_to_df(flooded_coors_dic, 1, region_name)
        all_flood_coors_df.append(flooded_coors_df)
        none_flooded_coors_df = coordinates_dict_to_df(none_flooded_coors_dic, 0, region_name)
        all_flood_coors_df.append(none_flooded_coors_df)

    # Concatenate all DataFrames into a single DataFrame and return.
    flood_samples = pd.concat(all_flood_coors_df)
    return flood_samples


def get_duration_values(input_df, save_path=None, scale=30):
    """
    Retrieves and appends the duration values for flood events to an input DataFrame.

    This function uses the Earth Engine to fetch 'duration' data for flood events based on the
    event ID and coordinates provided in the input DataFrame. The fetched 'duration' data is
    then appended to the input DataFrame as a new column.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood samples with columns 'Flood_ID', 'Lon', and 'Lat'.
    :type input_df: pd.DataFrame
    :param save_path: Path to save the resulting DataFrame with appended 'Duration' data. If None, the function won't save.
    :type save_path: str, optional
    :param scale: The spatial resolution of the flood data in meters.
    :type scale: float or int, default 30
    :return: Modified DataFrame containing the additional 'Duration' column.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> flood_data_df = pd.DataFrame({'Flood_ID': ['event1_1', 'event2_2'], 'Lon': [0, 5], 'Lat': [0, 5]})
        # >>> updated_df = get_duration_values(flood_data_df, save_path='updated_flood_data.csv')
    """

    # Create a copy of the input DataFrame to avoid altering the original data.
    flood_samples = input_df.copy()

    # Initialize Earth Engine
    ee.Initialize()

    # Load the global flood data from Google Earth Engine.
    global_flood_data = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

    # Initialize an empty list to store the features.
    features = []

    # Iterate over each row in the DataFrame to extract flood event ID and coordinates.
    for index, row in tqdm(flood_samples.iterrows(), total=flood_samples.shape[0], desc="Processing <Duration> column"):
        # Get the flood event ID from the DataFrame
        event_id = int(row['Flood_ID'].split('_')[-1])
        # Get the coordinate from the DataFrame
        coordinate = [row['Lon'], row['Lat']]

        # Create a feature for the given coordinate and associated flood event ID.
        feature = ee.Feature(ee.Geometry.Point(coordinate), {'event_id': event_id})
        # Append the feature to the list
        features.append(feature)

    # Convert the list of features into a FeatureCollection.
    feature_collection = ee.FeatureCollection(features)

    # Function to retrieve the duration value for a given flood event and coordinate.
    def get_duration(input_feature, input_scale=scale):
        # Get the event ID from the feature properties
        flood_event_id = input_feature.get('event_id')
        # Get the flood event
        flood_event = global_flood_data.filter(ee.Filter.eq('id', flood_event_id)).first()
        # Sample the 'duration' band at the input_feature's location and add the value to the input_feature properties
        return input_feature.set(flood_event.select('duration').reduceRegion(ee.Reducer.first(), input_feature.geometry(), input_scale))

    # Map the function over the FeatureCollection
    result_feature_collection = feature_collection.map(get_duration)

    # Extract the duration values from the resulting FeatureCollection.
    duration_values = result_feature_collection.aggregate_array('duration').getInfo()

    # Append the duration values to the DataFrame.
    flood_samples['Duration'] = duration_values

    # Save the updated DataFrame to a CSV file if a save_path is provided.
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    return flood_samples


def add_max_duration(input_df, save_path=None):
    """
    Appends the maximum duration value for each unique flood event to the input DataFrame.

    This function calculates and appends the maximum 'duration' data for each unique flood event
    (as denoted by 'Flood_ID') to the input DataFrame. The calculated max duration values are
    then appended to the input DataFrame as a new column called 'Max_Duration'.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood samples with columns 'Flood_ID' and 'Duration'.
    :type input_df: pd.DataFrame
    :param save_path: Path to save the resulting DataFrame with appended 'Max_Duration' data. If None, the function won't save.
    :type save_path: str, optional
    :return: Modified DataFrame containing the additional 'Max_Duration' column.
    :rtype: pd.DataFrame


    Example of usage:
    ----------------
        # >>> input_df = pd.DataFrame({
        #         'Flood_ID': ['event1', 'event1', 'event2', 'event2', 'event3'],
        #         'Duration': [2, 4, 1, 3, 5]
        #     })
        # >>> output_df = get_flood_sampling_data.add_max_duration(input_df)
    """

    # Create a copy of the input DataFrame to avoid altering the original data.
    flood_samples = input_df.copy()

    # Create a dictionary to store the maximum duration for each unique Flood_ID.
    max_duration_dict = {}

    # Get a list of unique Flood_IDs.
    unique_flood_ids = flood_samples['Flood_ID'].unique()

    # Iterate over each unique Flood_ID to calculate the max duration for that event.
    for flood_id in tqdm(unique_flood_ids, desc="Processing <Max_Duration> column"):
        # Get the subset of the DataFrame that corresponds to the current Flood_ID
        subset = flood_samples[flood_samples['Flood_ID'] == flood_id]
        # Get the maximum duration in the subset
        max_duration = subset['Duration'].max()
        # Add the maximum duration to the dictionary
        max_duration_dict[flood_id] = max_duration

    # Add a new column to the DataFrame with the maximum duration for each Flood_ID
    flood_samples['Max_Duration'] = flood_samples['Flood_ID'].map(max_duration_dict)

    # Save the DataFrame to a new CSV file if save_path is provided
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    return flood_samples


def add_event_start_and_end_date(input_df, historical_info_path, save_path=None):
    """
    Appends the start and end date of each flood event to the input DataFrame.

    The function calculates and appends the 'Event_Start_Date' and 'Event_End_Date'
    for each unique flood event (as denoted by 'Flood_ID') to the input DataFrame.
    The start date is fetched from the provided historical info table, and the end
    date is computed based on the start date and the 'Max_Duration' value.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood samples with 'Flood_ID' and 'Max_Duration' columns.
    :type input_df: pd.DataFrame
    :param historical_info_path: Path to the CSV file containing historical flood event information.
    :type historical_info_path: str
    :param save_path: Path to save the resulting DataFrame with appended date data. If None, the function won't save.
    :type save_path: str, optional
    :return: Modified DataFrame containing the additional 'Event_Start_Date' and 'Event_End_Date' columns.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> flood_data_df = pd.DataFrame({'Flood_ID': ['event1_1', 'event2_2'], 'Max_Duration': [5, 10]})
        # >>> historical_data_path = 'historical_flood_info.csv'
        # >>> updated_df = add_event_start_and_end_date(flood_data_df, historical_data_path, save_path='updated_date_data.csv')
    """

    # Create a copy of the input DataFrame to avoid altering the original data.
    flood_samples = input_df.copy()

    # Load the historical flood event information from the provided CSV file.
    historical_info_df = pd.read_csv(historical_info_path)

    # Create a dictionary to map event ID to start date
    event_start_date_dict = {str(row['id']): row['Start_Date'] for index, row in historical_info_df.iterrows()}

    # Function to extract last 4 digits of Flood_ID and map to start date
    def map_to_start_date(flood_id):
        event_id = flood_id.split('_')[-1]
        return event_start_date_dict.get(event_id, None)

    # Apply the function to the 'Flood_ID' column to populate the 'Event_Start_Date' column.
    tqdm.pandas(desc="Processing <Event_Start_Date> & <Event_End_Date> columns")
    flood_samples['Event_Start_Date'] = flood_samples['Flood_ID'].progress_apply(map_to_start_date)

    # Convert the 'Event_Start_Date' column to a datetime format.
    flood_samples['Event_Start_Date'] = pd.to_datetime(flood_samples['Event_Start_Date'])

    # Calculate the 'Event_End_Date' based on the 'Event_Start_Date' and 'Max_Duration' columns.
    flood_samples['Event_End_Date'] = flood_samples['Event_Start_Date'] + pd.to_timedelta(flood_samples['Max_Duration'], unit='d')

    # Save the DataFrame to a new CSV file if save_path is provided
    if save_path:
        flood_samples.to_csv(save_path, index=False)

    return flood_samples


def combine_duration_maxDuration_dates(input_df, historical_info_path, scale=30, save_path=None):
    """
    Calculate and append flood duration, maximum duration, start and end dates to the input DataFrame.

    This function combines three processes:
    1. Getting the duration values for each flood sample,
    2. Calculating the maximum duration for each flood event, and
    3. Appending the start and end date of each flood event.


    Parameters:
    -----------
    :param input_df: DataFrame containing flood samples.
    :type input_df: pd.DataFrame
    :param historical_info_path: Path to the CSV file containing historical flood event information.
    :type historical_info_path: str
    :param scale: The spatial resolution of the flood data in meters, default is 30.
    :type scale: float or int, optional
    :param save_path: Path to save the resulting DataFrame with appended data. If None, the function won't save.
    :type save_path: str, optional
    :return: Modified DataFrame containing the additional 'Duration', 'Max_Duration', 'Event_Start_Date', and 'Event_End_Date' columns.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> flood_data_df = pd.DataFrame({'Flood_ID': ['event1_1', 'event2_2']})
        # >>> historical_data_path = 'historical_flood_info.csv'
        # >>> updated_df = combine_duration_maxDuration_dates(flood_data_df, historical_data_path, save_path='updated_data.csv')
    """

    # Create a copy of the input DataFrame to avoid altering the original data.
    flood_samples = input_df.copy()

    # Fetch the duration values for each flood sample.
    flood_samples = get_duration_values(flood_samples, scale=scale)

    # Calculate the maximum duration for each unique flood event.
    flood_samples = add_max_duration(flood_samples)

    # Append the start and end date for each flood event.
    flood_samples = add_event_start_and_end_date(flood_samples, historical_info_path)

    # Save the DataFrame to a new CSV file if save_path is provided
    if save_path:
        flood_samples.to_csv(save_path, index=False)
    return flood_samples


def sample_in_blocks(roi, block_size_degree, image, scale):
    """
    Sample an image within blocks of a specified size in the given region of interest (ROI).

    This function divides a given ROI into rectangular blocks of a specified size in degrees.
    Within each block, the image is sampled at a specified scale. The resulting sample points from
    all blocks are then combined into a single FeatureCollection.


    Parameters:
    -----------
    :param roi: The region of interest where the image will be sampled.
    :type roi: ee.Geometry or ee.Feature or ee.FeatureCollection
    :param block_size_degree: The size of each rectangular block in degrees.
    :type block_size_degree: float
    :param image: The Earth Engine image to be sampled.
    :type image: ee.Image
    :param scale: The spatial resolution at which the image will be sampled, in meters.
    :type scale: float or int
    :return: A FeatureCollection containing the sample points from all blocks.
    :rtype: ee.FeatureCollection


    Example of usage:
    -----------------
        # >>> img = ee.Image("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
        # >>> roi1 = ee.Geometry.Rectangle([-122.5, 37.7, -121.8, 37.9])
        # >>> samples = sample_in_blocks(roi1, 0.1, img, 30)
    """

    # Fetch the boundaries of the region of interest
    bounds = roi.bounds().coordinates().get(0).getInfo()
    # Extract the x and y bounds
    xmin, ymin, xmax, ymax = bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1]

    # Generate list of x and y coordinates for the blocks
    xs = list(np.arange(xmin, xmax, block_size_degree))
    ys = list(np.arange(ymin, ymax, block_size_degree))

    results = []

    # Iterate over x and y coordinates
    for x in xs:
        for y in ys:
            # Define block boundaries
            block_xmax = x + block_size_degree if x + block_size_degree <= xmax else xmax
            block_ymax = y + block_size_degree if y + block_size_degree <= ymax else ymax
            # Create a block geometry
            block = ee.Geometry.Rectangle([x, y, block_xmax, block_ymax])
            # Clip the image to the block boundaries
            clipped_image = image.clip(block)
            # Sample the clipped image
            sampled_points = clipped_image.sample(scale=scale, geometries=True)
            results.append(sampled_points)

    # Combine results from all blocks into a single FeatureCollection
    return ee.FeatureCollection(results).flatten()


def uniform_get_flood_data_from_ROI(history_flood_event_ids, ROI_boundary, scale=250, block_size_degree=0.05):
    """
    Uniformly sample flood data from a Region Of Interest (ROI) for historical flood events.

    This function retrieves flood data for a list of historical flood event IDs within a specified ROI.
    The sampling is done uniformly within blocks of a given size. The function returns a dictionary where
    each key is a flood event ID and the corresponding value is a list of sampled points with their flood labels.


    Parameters:
    -----------
    :param history_flood_event_ids: List of historical flood event IDs to be sampled.
    :type history_flood_event_ids: list[int]
    :param ROI_boundary: The boundary of the Region Of Interest (ROI) where the data will be sampled.
    :type ROI_boundary: ee.Geometry or ee.Feature or ee.FeatureCollection
    :param scale: The spatial resolution at which the data will be sampled, in meters.
    :type scale: float or int
    :param block_size_degree: The size of each rectangular block in degrees for uniform sampling.
    :type block_size_degree: float
    :return: A dictionary where each key is a flood event ID and the value is a list of sampled points.
    :rtype: dict[int, list[list[float, float, int]]]


    Example of usage:
    -----------------
        # >>> ids = [1001, 1002, 1003]
        # >>> roi = ee.Geometry.Rectangle([-122.5, 37.7, -121.8, 37.9])
        # >>> data = uniform_get_flood_data_from_ROI(ids, roi)
    """

    # Initialize the Earth Engine collection for global flood data
    global_flood_data = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
    data_dict = {}

    # Iterate over the provided list of flood event IDs with a progress bar
    for event_id in tqdm(history_flood_event_ids, desc="Sampling data from each event"):
        # Retrieve the specific flood event image using the event ID
        flood_event = global_flood_data.filter(ee.Filter.eq('id', event_id)).first()
        # Define the expression to identify flooded areas that are not permanent water bodies
        flooded_and_not_perm_water = flood_event.expression(
            "(flooded == 1) && (jrc_perm_water == 0)",
            {
                'flooded': flood_event.select('flooded'),
                'jrc_perm_water': flood_event.select('jrc_perm_water')
            }
        ).rename("label")

        # Use the sample_in_blocks function to sample the flood data uniformly within blocks
        points = sample_in_blocks(ROI_boundary, block_size_degree, flooded_and_not_perm_water, scale)
        features = points.getInfo()['features']

        event_data = []
        # Extract coordinates and flood labels from the sampled points
        for feature in features:
            coordinates = feature['geometry']['coordinates']
            label = feature['properties']['label']
            event_data.append([coordinates[0], coordinates[1], label])

        # Store the sampled data for the current event ID in the dictionary
        data_dict[event_id] = event_data

    return data_dict