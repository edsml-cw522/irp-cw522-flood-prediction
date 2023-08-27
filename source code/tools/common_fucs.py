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


def check_duplicate_rows(input_df, columns=None, print_info=False):
    """
    Checks a DataFrame for duplicate rows based on either the entire row or based on specified columns row.

    The function determines if there are any duplicate rows in the input DataFrame.
    If specific columns are provided, it will only check for duplicate rows in these columns.
    Otherwise, it will consider the entire row for duplication checks.


    Parameters:
    -----------
    :param input_df: DataFrame to be checked for duplicates.
    :type input_df: pd.DataFrame
    :param columns: List of columns to be checked for duplicates. If None, the entire row will be considered.
    :type columns: list, optional
    :param print_info: Flag to print the duplicate rows, if found.
    :type print_info: bool, default False
    :return: Boolean value indicating the presence of duplicates. True if duplicates exist, False otherwise.
    :rtype: bool


    Example of usage:
    ----------------
        # >>> import pandas as pd
        # >>> df1 = pd.DataFrame({'A': [1, 2, 3, 2], 'B': [5, 6, 7, 6]})
        # >>> check_duplicate_rows(df1)
        # True
        # >>> check_duplicate_rows(df1, columns=['A'])
        # True
    """
    # Create a copy of the input DataFrame to avoid any in-place modifications
    df = input_df.copy()

    # If specific columns are provided, check only those columns for duplicates
    if columns:
        duplicate_rows = df.duplicated(subset=columns, keep=False)
        message = f"Rows formed by columns {columns} have no duplicates."
    # Otherwise, check the entire row for duplicates
    else:
        duplicate_rows = df.duplicated(keep=False)
        message = "The entire rows have no duplicates."

    # If the 'print_info' flag is True, print information about duplicate rows
    if duplicate_rows.any():
        if print_info:
            print("Found duplicate rows:")
            print(df[duplicate_rows])
        return True
    else:
        if print_info:
            print(message)
        return False


def discard_duplicates(input_df, columns=None, print_info=False):
    """
    Discards duplicate rows in a DataFrame based on either the entire row or based on specified columns row.

    The function removes any duplicate rows in the input DataFrame. If specific columns are provided,
    it will only consider duplicates in these columns. Otherwise, it will consider the entire row for
    duplication checks. By default, the first occurrence of a duplicate is kept and subsequent occurrences are discarded.


    Parameters:
    -----------
    :param input_df: DataFrame from which duplicates are to be removed.
    :type input_df: pd.DataFrame
    :param columns: List of columns to be considered for checking duplicates. If None, the entire row will be considered.
    :type columns: list, optional
    :param print_info: Flag to print information about the removal of duplicates.
    :type print_info: bool, default False
    :return: DataFrame with duplicates removed.
    :rtype: pd.DataFrame


    Example of usage:
    -----------------
        # >>> import pandas as pd
        # >>> df1 = pd.DataFrame({'A': [1, 2, 3, 2], 'B': [5, 6, 7, 6]})
        # >>> new_df = discard_duplicates(df)
        # >>> new_df
        #    A  B
        # 0  1  5
        # 1  2  6
        # 2  3  7
    """
    # Create a copy of the input DataFrame to avoid any in-place modifications
    df = input_df.copy()
    # If specific columns are provided, check only those columns for duplicates
    if columns:
        duplicates = df.duplicated(subset=columns, keep='first')
    # Otherwise, check the entire row for duplicates
    else:
        duplicates = df.duplicated(keep='first')

    # If duplicates are found and the 'print_info' flag is True, print information
    # and discard the duplicates
    if duplicates.any():
        if print_info:
            print("Discarding duplicate rows...")
        df = df.loc[~duplicates]
    else:
        if print_info:
            print("No duplicate rows found. Returning the original DataFrame.")
    return df


def compare_float_lists(list1, list2, tolerance=1e-7):
    """
    Compare two lists of floating-point numbers for approximate equality.

    The function checks if two lists of floats are approximately equal, element by element,
    within a specified tolerance. The lengths of the lists must also match for them to be
    considered equal.


    Parameters:
    -----------
    :param list1: The first list of floating-point numbers.
    :type list1: list[float]
    :param list2: The second list of floating-point numbers.
    :type list2: list[float]
    :param tolerance: The maximum allowable difference between corresponding elements for them to be considered equal.
    :type tolerance: float, default 1e-7
    :return: True if the lists are approximately equal, False otherwise.
    :rtype: bool


    Example of usage:
    -----------------
        # >>> list_a = [0.1 + 0.2, 0.3]
        # >>> list_b = [0.3, 0.3]
        # >>> compare_float_lists(list_a, list_b)
        # True
    """
    # If the lengths of the lists are not the same, they are not equal
    if len(list1) != len(list2):
        return False

    # Compare each pair of elements
    for a, b in zip(list1, list2):
        # If the absolute difference between the elements is greater than the specified tolerance, they are not equal
        if abs(a - b) > tolerance:
            return False
    # If all pairs of elements are approximately equal, the lists are considered equal
    return True


def compare_float_2d_lists(list1, list2, tolerance=1e-7):
    """
    Compare two 2D lists (lists of lists) of floating-point numbers for approximate equality.

    This function checks if two 2D lists of floats are approximately equal, sublist by sublist,
    and element by element within each sublist, within a specified tolerance.
    The lengths of the outer lists and inner sublists must match for the entire 2D lists to be considered equal.
    Floating-point precision issues, which might cause minor discrepancies between values, are accounted for.


    Parameters:
    -----------
    :param list1: The first 2D list of floating-point numbers.
    :type list1: list[list[float]]
    :param list2: The second 2D list of floating-point numbers.
    :type list2: list[list[float]]
    :param tolerance: The maximum allowable difference between corresponding elements for them to be considered equal.
    :type tolerance: float, default 1e-7
    :return: True if the 2D lists are approximately equal, False otherwise.
    :rtype: bool


    Example of usage:
    ----------------
        # >>> list_a_2d = [[0.1 + 0.2, 0.3], [0.4, 0.5]]
        # >>> list_b_2d = [[0.3, 0.3], [0.4, 0.5]]
        # >>> compare_float_2d_lists(list_a_2d, list_b_2d)
        # True
    """

    # If the lengths of the outer lists are not the same, they are not equal
    if len(list1) != len(list2):
        return False
    # Compare each pair of 1D sublist
    for sublist1, sublist2 in zip(list1, list2):
        # Use the previously defined compare_float_lists function to compare each pair of 1D sublist
        if not compare_float_lists(sublist1, sublist2, tolerance):
            return False
    # If all pairs of 1D sublist are approximately equal, the 2D lists are considered equal
    return True


def convert_to_ee_rectangle(coordinates):
    """
    Convert a list of coordinates into an Earth Engine Rectangle object.

    This function takes a list of coordinates, extracts the minimum and maximum longitude
    and latitude values, and constructs an Earth Engine Rectangle object from these values.
    The input is expected to be a list containing a list of coordinate pairs (longitude, latitude).


    Parameters:
    -----------
    :param coordinates: A list containing a list of (longitude, latitude) pairs.
    :type coordinates: list[list[tuple[float, float]]]
    :return: An Earth Engine Rectangle object constructed from the given coordinates.
    :rtype: ee.Geometry.Rectangle


    Example of usage:
    -----------------
        # >>> coords1 = [[[-0.642576, 8.172062], [-0.502478, 8.172062], [-0.502478, 8.339226], [-0.642576, 8.339226], [-0.642576, 8.172062]]]
        # >>> rectangle1 = convert_to_ee_rectangle(coords1)
    """
    # Extract the minimum and maximum longitude and latitude values from the coordinates
    min_lon = min(coord[0] for coord in coordinates[0])
    max_lon = max(coord[0] for coord in coordinates[0])
    min_lat = min(coord[1] for coord in coordinates[0])
    max_lat = max(coord[1] for coord in coordinates[0])

    # Create an Earth Engine Rectangle object using the extracted values
    rectangle = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    return rectangle
