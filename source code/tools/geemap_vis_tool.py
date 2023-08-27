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
import geemap
import ee


def load_flood_data_from_csv(filepath):
    """
    Load flood data from a CSV file.

    This function reads a CSV file containing flood data and returns a DataFrame
    with selected columns: "Flood_ID", "Lon", "Lat", and "label".


    Parameters:
    -----------
    :param filepath: The path to the CSV file.
    :type filepath: str
    :return: DataFrame containing flood data.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> df = load_flood_data_from_csv("flood_data.csv")
    """
    df = pd.read_csv(filepath)
    return df[["Flood_ID", "Lon", "Lat", "label"]]


def create_boundary_mask(boundary_gee):
    """
    Create a mask from the given boundary using Google Earth Engine.

    This function creates a mask based on the boundary provided. The mask can be
    used for visualization purposes to outline a specific region.


    Parameters:
    -----------
    :param boundary_gee: The boundary (either a Geometry or FeatureCollection) for which the mask is to be created.
    :type boundary_gee: ee.geometry.Geometry or ee.FeatureCollection
    :return: Image containing the mask.
    :rtype: ee.Image


    Example of usage:
    -----------------
        # >>> boundary = ee.Geometry.Polygon([...])
        # >>> mask_image = create_boundary_mask(boundary)
    """
    empty = ee.Image().byte()

    # Check if the input is a Geometry or FeatureCollection
    if isinstance(boundary_gee, ee.geometry.Geometry):
        boundary_gee = ee.FeatureCollection([ee.Feature(boundary_gee)])

    # Define the visualization parameters.
    outline = empty.paint(**{
        'featureCollection': boundary_gee,
        'color': 1,
        'width': 3
    })
    mask = ee.Image(0).mask(0)
    mask = outline.blend(mask)
    return mask


def df_to_ee_point_features(df):
    """
    Convert a pandas DataFrame to a Google Earth Engine (GEE) FeatureCollection of point features.

    This function transforms each row of the DataFrame into a GEE point feature,
    using the "Lon" and "Lat" columns for the point's coordinates and the "label"
    column for an associated property. The resulting point features are then
    collected into a GEE FeatureCollection.


    Parameters:
    -----------
    :param df: The DataFrame containing columns "Lon", "Lat", and "label".
    :type df: pd.DataFrame
    :return: A FeatureCollection containing point features derived from the DataFrame.
    :rtype: ee.FeatureCollection


    Example of usage:
    -----------------
        # >>> df = pd.DataFrame({
        # ...     'Lon': [10, 20, 30],
        # ...     'Lat': [-10, -20, -30],
        # ...     'label': ['A', 'B', 'C']
        # ... })
        # >>> point_features = df_to_ee_point_features(df)
    """
    features = []
    for _, row in df.iterrows():
        point = ee.Geometry.Point([row["Lon"], row["Lat"]])
        feature = ee.Feature(point, {"label": row["label"]})
        features.append(feature)
    return ee.FeatureCollection(features)


def display_data_on_map(history_flood_event_ids, flooded_and_not_perm_water_viz_params, jrc_perm_water_viz_params,
                        duration_viz_params, mask_dic, df, save_html_map_path=None):
    """
        Display flood data on an interactive map.

        This function creates an interactive map using the geemap library and overlays various flood-related
        datasets from the Earth Engine data catalog. The function also allows for the visualization of custom
        flood-related data stored in a pandas DataFrame.


        Parameters:
        -----------
        :param history_flood_event_ids: List of historical flood event IDs to be visualized.
        :type history_flood_event_ids: list[int]
        :param flooded_and_not_perm_water_viz_params: Visualization parameters for areas flooded excluding permanent water bodies.
        :type flooded_and_not_perm_water_viz_params: dict
        :param jrc_perm_water_viz_params: Visualization parameters for JRC permanent water sources.
        :type jrc_perm_water_viz_params: dict
        :param duration_viz_params: Visualization parameters for flood duration.
        :type duration_viz_params: dict
        :param mask_dic: Dictionary containing masks or boundary layers for visualization.
        :type mask_dic: dict
        :param df: DataFrame containing flood data with columns "Flood_ID", "Lon", "Lat", and "label".
        :type df: pd.DataFrame
        :param save_html_map_path: Path to save the interactive map as an HTML file. If None, the map is not saved.
        :type save_html_map_path: str, optional
        :return: The interactive map object.
        :rtype: geemap.Map


        Example of usage:
        -----------------
            # >>> history_flood_event_ids = [12345, 67890]
            # >>> flooded_and_not_perm_water_viz_params = {'palette': 'blue'}
            # >>> jrc_perm_water_viz_params = {'palette': 'green'}
            # >>> duration_viz_params = {'palette': 'red'}
            # >>> mask_dic = {"boundary": ee.Image().byte()}
            # >>> df = load_flood_data_from_csv("flood_data.csv")
            # >>> Map = display_data_on_map(history_flood_event_ids, flooded_and_not_perm_water_viz_params,
            # ...                           jrc_perm_water_viz_params, duration_viz_params, mask_dic, df)
        """

    # Initialize the Earth Engine module
    ee.Initialize()

    # Center the map over Ghana
    center = [7.9465, -1.0232]  # Latitude, Longitude for Ghana
    Map = geemap.Map(center=center, zoom=6)

    # get flood data:
    global_flood_data = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")

    # Add layers for each historical flood event
    for event_id in history_flood_event_ids:
        flood_event = global_flood_data.filter(ee.Filter.eq('id', event_id)).first()

        # Obtain the area that was actually flooded (areas where permanent bodies of water were excluded)
        flooded_and_not_perm_water = flood_event.expression(
            "(flooded == 1) && (jrc_perm_water == 0)",
            {
                'flooded': flood_event.select('flooded'),
                'jrc_perm_water': flood_event.select('jrc_perm_water')
            }
        )

        # Show the permanent water source and duration in each event
        jrc_perm_water = flood_event.select('jrc_perm_water')
        duration = flood_event.select('duration')
        Map.addLayer(jrc_perm_water, jrc_perm_water_viz_params, f'perm_water {event_id}')
        Map.addLayer(duration, duration_viz_params, f'duration {event_id}')

        # Add a layer of the actual flooded area
        Map.addLayer(flooded_and_not_perm_water, flooded_and_not_perm_water_viz_params,
                     f'Actual Flooded Area {event_id}')

    for key, value in mask_dic.items():
        # Add boundary layers:
        Map.addLayer(value, {'palette': '000000'}, key)

    # Group by Flood_ID and convert each group to an ee.FeatureCollection
    grouped = df.groupby("Flood_ID")
    flood_feature_collections = {flood_id: df_to_ee_point_features(group_df) for flood_id, group_df in grouped}

    for flood_id, feature_collection in flood_feature_collections.items():
        vis_params_label_0 = {'color': '#b3a402'}
        vis_params_label_1 = {'color': '#32ede4'}

        # Filter features based on label value
        features_label_0 = feature_collection.filter(ee.Filter.eq('label', 0))
        features_label_1 = feature_collection.filter(ee.Filter.eq('label', 1))

        # Add layers of sampled points
        Map.addLayer(features_label_0, vis_params_label_0, f"{flood_id} - label 0")
        Map.addLayer(features_label_1, vis_params_label_1, f"{flood_id} - label 1")

    # Display the map
    Map.addLayerControl()
    display(Map)
    if save_html_map_path:
        Map.save(outfile=save_html_map_path)
    return Map