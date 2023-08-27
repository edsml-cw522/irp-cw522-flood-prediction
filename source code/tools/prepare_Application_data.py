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
from get_flood_sampling_data import combine_duration_maxDuration_dates
from get_precipitation_data import download_daily_precip_dic, combine_precip
from common_fucs import compare_float_lists, discard_duplicates
import pandas as pd


def auto_get_flood_data_for_application(input_df,
                                        historical_info_path,
                                        daily_precip_pickle_file_path,
                                        geo_data_path,
                                        download_daily_precip=False,
                                        save_path=None):
    """
    Automatically retrieve and process flood data for application purposes.

    This function processes input flood data, optionally downloads daily precipitation data,
    and merges it with geographic data for the same samples. The function is designed for
    application use cases where data needs to be fetched, processed, and returned in a structured format.


    Parameters:
    -----------
    :param input_df: Input DataFrame containing flood sample data.
    :type input_df: pd.DataFrame
    :param historical_info_path: Path to the historical flood information.
    :type historical_info_path: str
    :param daily_precip_pickle_file_path: Path to the pickle file where daily rainfall data is saved.
    :type daily_precip_pickle_file_path: str
    :param geo_data_path: Path to the flood samples geographic data.
    :type geo_data_path: str
    :param download_daily_precip: Whether to download daily precipitation data, default is False.
    :type download_daily_precip: bool, default False
    :param save_path: Where to save the resulting DataFrame as a CSV file, if provided.
    :type save_path: str or None, default None
    :return: A DataFrame containing the processed flood data for application.
    :rtype: pd.DataFrame


    Example of usage:
    ----------------
        # >>> df = auto_get_flood_data_for_application(input_df=pd.DataFrame(...),
        #                                         historical_info_path='path/to/historical_info.csv',
        #                                         daily_precip_pickle_file_path='path/to/rainfall_data.pkl',
        #                                         geo_data_path='path/to/geo_data.csv',
        #                                         download_daily_precip=True)
    """

    flood_samples_ls = []
    flood_samples = input_df.copy()

    # If specified, download daily precipitation data for each flood ID group
    if download_daily_precip:
        for flood_id, group in flood_samples.groupby('Flood_ID'):
            print(f"Processing {flood_id}:")
            data = combine_duration_maxDuration_dates(group, historical_info_path)
            _ = download_daily_precip_dic(data, save_path=daily_precip_pickle_file_path+str(flood_id)+str(".pickle"))
            print()
            print()

    # Process each flood ID group to combine precipitation related columns
    for flood_id, group in flood_samples.groupby('Flood_ID'):
        print(f"Processing {flood_id}:")
        data = combine_precip(group, daily_precip_pickle_file_path+str(flood_id)+str(".pickle"))
        flood_samples_ls.append(data)

    flood_samples = pd.concat(flood_samples_ls)
    geo_data = pd.read_csv(geo_data_path)

    # Compare and merge dataframes based on coordinates and flood state
    flood_samples_Lon = flood_samples['Lon'].tolist()
    flood_samples_geo_data_Lon = geo_data['Lon'].tolist()
    flood_samples_Lat = flood_samples['Lat'].tolist()
    flood_samples_geo_data_Lat = geo_data['Lat'].tolist()
    flood_samples_Flooded = flood_samples['Flooded'].tolist()
    flood_samples_geo_data_label = geo_data['label'].tolist()

    # Drop repeated columns if the values match:
    if compare_float_lists(flood_samples_Lon, flood_samples_geo_data_Lon):
        geo_data.drop(columns=['Lon'], inplace=True)

    if compare_float_lists(flood_samples_Lat, flood_samples_geo_data_Lat):
        geo_data.drop(columns=['Lat'], inplace=True)

    if compare_float_lists(flood_samples_Flooded, flood_samples_geo_data_label):
        flood_samples.reset_index(drop=True, inplace=True)
        geo_data.reset_index(drop=True, inplace=True)
        flood_samples = pd.concat([flood_samples, geo_data], axis=1)
        flood_samples.drop(columns=['Flooded'], inplace=True)

    # Remove duplicates
    flood_samples_for_application = discard_duplicates(flood_samples, print_info=True)

    # Save the processed DataFrame if a save path is provided
    if save_path:
        flood_samples_for_application.to_csv(save_path, index=False)
        print(f"your file has been saved to {save_path} ")

    return flood_samples_for_application
