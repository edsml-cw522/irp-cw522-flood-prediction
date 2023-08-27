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
from get_flood_sampling_data import auto_get_flood_data_df_from_a_list_ROIs
from get_flood_sampling_data import combine_duration_maxDuration_dates
from get_precipitation_data import combine_precip, download_daily_precip_dic
from common_fucs import discard_duplicates, compare_float_lists
import pandas as pd


def get_flood_data(flood_ids, historical_info_path,
                   flood_samples_geo_data_path,
                   daily_precip_pickle_file_path,
                   ROI_features_ls_3d, flood_data_scale=30,
                   download_daily_precip = False,
                   download_daily_precip_batch_size = 1000,
                   download_daily_precip_scale = 5566,
                   check_each_sample_amount=False, print_info=True, save_path=None):
    """
    Retrieve and process flood data for a set of flood event IDs.

    This function automatically gets flood data for a list of Region Of Interests (ROIs),
    combines duration, maximum duration, start and end dates, and merges it with geographic
    data for the same samples. The function also has an option to download daily precipitation data.


    Parameters:
    -----------
    :param flood_ids: List of flood event IDs to retrieve data for.
    :type flood_ids: list[int]
    :param historical_info_path: Path to the historical flood information.
    :type historical_info_path: str
    :param flood_samples_geo_data_path: Path to the flood samples geographic data.
    :type flood_samples_geo_data_path: str
    :param daily_precip_pickle_file_path: Path to the pickle file containing daily rainfall data.
    :type daily_precip_pickle_file_path: str
    :param ROI_features_ls_3d: List of Region Of Interests (ROIs) features in 3D.
    :type ROI_features_ls_3d: list
    :param flood_data_scale: Spatial resolution of the flood data in meters, default is 30.
    :type flood_data_scale: int, default 30
    :param download_daily_precip: Whether to download daily precipitation data, default is False.
    :type download_daily_precip: bool, default False
    :param download_daily_precip_batch_size: Batch size for downloading daily precipitation data.
    :type download_daily_precip_batch_size: int, default 1000
    :param download_daily_precip_scale: Spatial resolution for downloading daily precipitation data.
    :type download_daily_precip_scale: int, default 5566
    :param check_each_sample_amount: Whether to check the amount for each sample, default is False.
    :type check_each_sample_amount: bool, default False
    :param print_info: Whether to print information during processing, default is True.
    :type print_info: bool, default True
    :param save_path: Where to save the resulting DataFrame as a CSV file, if provided.
    :type save_path: str or None, default None
    :return: A DataFrame containing the processed flood data.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = get_flood_data(flood_ids=[1,2,3], historical_info_path='path/to/historical_info.csv',
        #                         flood_samples_geo_data_path='path/to/geo_data.csv',
        #                         daily_precip_pickle_file_path='path/to/rainfall_data.pkl',
        #                         ROI_features_ls_3d=[[0,0,5,5]])
    """

    # Get flood samples data from a list of ROIs
    flood_samples = auto_get_flood_data_df_from_a_list_ROIs(flood_ids,
                                                            ROI_features_ls_3d,
                                                            scale=flood_data_scale,
                                                            check_each_sample_amount=check_each_sample_amount,
                                                            print_info=print_info)

    # Combine duration, max duration, start and end dates
    flood_samples = combine_duration_maxDuration_dates(flood_samples,
                                              historical_info_path = historical_info_path,
                                              scale=flood_data_scale)

    # Download daily precipitation data if specified
    if download_daily_precip:
        download_daily_precip_dic(input_df=flood_samples,
                                  save_path = daily_precip_pickle_file_path,
                                  batch_size=download_daily_precip_batch_size,
                                  scale=download_daily_precip_scale)

    # Add precipitation related columns
    flood_samples = combine_precip(flood_samples, daily_precip_pickle_file_path)

    # Load flood samples geographic data
    flood_samples_geo_data = pd.read_csv(flood_samples_geo_data_path)
    flood_samples_geo_data.drop(columns=['Unnamed: 0'], inplace=True)

    # Compare and merge dataframes based on coordinates and flood state
    flood_samples_Lon = flood_samples['Lon'].tolist()
    flood_samples_geo_data_Lon = flood_samples_geo_data['Lon'].tolist()
    flood_samples_Lat = flood_samples['Lat'].tolist()
    flood_samples_geo_data_Lat = flood_samples_geo_data['Lat'].tolist()
    flood_samples_Flooded = flood_samples['Flooded'].tolist()
    flood_samples_geo_data_label = flood_samples_geo_data['label'].tolist()

    # Drop repeated columns if the values match
    if compare_float_lists(flood_samples_Lon, flood_samples_geo_data_Lon):
        flood_samples_geo_data.drop(columns=['Lon'], inplace=True)

    if compare_float_lists(flood_samples_Lat, flood_samples_geo_data_Lat):
        flood_samples_geo_data.drop(columns=['Lat'], inplace=True)

    if compare_float_lists(flood_samples_Flooded, flood_samples_geo_data_label):
        flood_samples.reset_index(drop=True, inplace=True)
        flood_samples_geo_data.reset_index(drop=True, inplace=True)
        flood_samples = pd.concat([flood_samples, flood_samples_geo_data], axis=1)
        flood_samples.drop(columns=['Flooded'], inplace=True)

    # Remove duplicates
    flood_samples_for_Modeling = discard_duplicates(flood_samples, print_info=True)

    # Save the processed DataFrame if a save path is provided
    if save_path:
        flood_samples_for_Modeling.to_csv(save_path, index=False)
        print(f"your file has been saved to {save_path} ")

    return flood_samples_for_Modeling
