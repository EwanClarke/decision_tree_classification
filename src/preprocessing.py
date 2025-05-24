# primary objective of preprocessing is the discretisation of continuous data types
from src.datasetSplitting import split_list_values, calculate_sublist_sizes

# data binning for continuous values
def equal_frequency_data_binning(no_of_bins:int, column):
    sorted_values = sorted(column)
    sublist_sizes = calculate_sublist_sizes(no_of_bins, len(sorted_values))
    bins = split_list_values(sublist_sizes, sorted_values)

    lowest_values = [specific_bin[0] for specific_bin in bins]
    binned_column = []
    for value in column:
        for i in range(no_of_bins):
            if i == 0 and value < lowest_values[1]:
                binned_column.append(f"<{lowest_values[1]}")
                break
            elif i == no_of_bins-1:
                binned_column.append(f">={lowest_values[-1]}")
            elif lowest_values[i] <= value < lowest_values[i + 1]:
                binned_column.append(f"{lowest_values[i]}-{lowest_values[i+1]}")
                break
    return binned_column
