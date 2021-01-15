import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import copy
import math
from tabulate import tabulate
from operator import itemgetter


def discretization(table, index, num_bins):
    # discretization
    # converting a numeric (continuous) attribute to discrete (categorical)
    # we will implement equal width bin discretization
    for row in table:
        row[index] = float(row[index])

    table = sorted(table, key=itemgetter(index))
    prices = get_column(table, index)
    values = prices
    cutoffs = compute_equal_widths_cutoffs(values, num_bins)
#     print("cutoffs:", cutoffs)
    np_freqs, np_cutoffs = np.histogram(values, num_bins)
#     print("np_cutoffs:", np_cutoffs[1:])
    
    # bins defined by cutoffs
    freqs = compute_bin_frequencies(values, cutoffs)
#     print("freqs:", freqs)
#     print("np_freqs:", np_freqs)
    num = 0
    


    # replace prices with discretited cutoff 1-10
    for row in table:
        if num_bins < 9:
            if float(row[index]) < cutoffs[0] or float(row[index]) == cutoffs[0]:
                row[index] = 1
            elif float(row[index]) == cutoffs[1] or (float(row[index]) > cutoffs[0] and float(row[index]) < cutoffs[1]):
                row[index] = 2
            elif (float(row[index]) > cutoffs[1] and float(row[index]) < cutoffs[2]) or float(row[index]) == cutoffs[2]:
                row[index] = 3
            elif (float(row[index]) > cutoffs[2] and float(row[index]) < cutoffs[3]) or float(row[index]) == cutoffs[3]:
                row[index] = 4
            elif (float(row[index]) > cutoffs[3] and float(row[index]) < cutoffs[4]) or float(row[index]) == cutoffs[4]:
                row[index] = 5      
        else:
            if float(row[index]) < cutoffs[0] or float(row[index]) == cutoffs[0]:
                row[index] = 1
            elif float(row[index]) == cutoffs[1] or (float(row[index]) > cutoffs[0] and float(row[index]) < cutoffs[1]):
                row[index] = 2
            elif (float(row[index]) > cutoffs[1] and float(row[index]) < cutoffs[2]) or float(row[index]) == cutoffs[2]:
                row[index] = 3
            elif (float(row[index]) > cutoffs[2] and float(row[index]) < cutoffs[3]) or float(row[index]) == cutoffs[3]:
                row[index] = 4
            elif (float(row[index]) > cutoffs[3] and float(row[index]) < cutoffs[4]) or float(row[index]) == cutoffs[4]:
                row[index] = 5 
            elif (float(row[index]) > cutoffs[4] and float(row[index]) < cutoffs[5]) or float(row[index]) == cutoffs[5]:
                row[index] = 6
            elif (float(row[index]) > cutoffs[5] and float(row[index]) < cutoffs[6]) or float(row[index]) == cutoffs[6]:
                row[index] = 7
            elif (float(row[index]) > cutoffs[6] and float(row[index]) < cutoffs[7]) or float(row[index]) == cutoffs[7]:
                row[index] = 8
            elif (float(row[index]) > cutoffs[7] and float(row[index]) < cutoffs[8]) or float(row[index]) == cutoffs[8]:
                row[index] = 9
            elif (float(row[index]) > cutoffs[8] and float(row[index]) < cutoffs[9]) or float(row[index]) == cutoffs[9]:
                row[index] = 10
                    
    return table

def read_attribute(table, index):
    my_list = []
    for row in table:
        my_list.append(float(row[index]))

    return my_list

def get_top_k(row_distances, k):
    row_distances.sort()
    top_k = row_distances[:k]

    return top_k


def summary_stats(my_list):
    mean = 0
    std = 0
    
    array = np.array(my_list)

    mean = round(np.mean(array),2)
    std = round(np.std(array),2)

    return mean, std

def get_summary_stats(table):
    table2 = copy.deepcopy(table)
    mean_list = []
    std_list = []
    att_list = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x_length', 'y_width', 'z_depth']
    # 0, 4-9
    att_list_print = ['carat', 'depth', 'table', 'price', 'x_length', 'y_width', 'z_depth']
    
    mean, std = summary_stats(get_column(table2, 0))
    mean_list.append(mean)
    std_list.append(std)
    
    for i in range(4, len( att_list)):
        mean, std = summary_stats(get_column(table2, i))
        mean_list.append(mean)
        std_list.append(std)
        
    my_table = [["attribute: " + att_list_print[0]], ["mean: " + str(mean_list[0])], ["std: " + str(std_list[0])]]
    print(tabulate(my_table, headers=[
          "SUMMARY STATISTICS: "], tablefmt="fancy_grid"))

    for i in range(1, 6):
        my_table = [["attribute: " + att_list_print[i]], ["mean: " + str(mean_list[i])], ["std: " + str(std_list[i])]]
        print(tabulate(my_table, tablefmt="fancy_grid"))


def cut_continuous(table): # index 1
    for row in table:
        if row[1] == "Fair":
            row[1] = 1
        elif row[1] == "Good":
            row[1] = 2
        elif row[1] == "Very Good":
            row[1] = 3
        elif row[1] == "Premium":
            row[1] = 4
        elif row[1] == "Ideal":
            row[1] = 5
    
    return table

def color_continuous(table): # index 2
    for row in table: # reveresed the order. Before a "higher value" or high letter in the alphabet was was worse
        if row[2] == "D":
            row[2] = 7
        elif row[2] == "E":
            row[2] = 6
        elif row[2] == "F":
            row[2] = 5
        elif row[2] == "G":
            row[2] = 4
        elif row[2] == "H":
            row[2] = 3
        elif row[2] == "I":
            row[2] = 2
        elif row[2] == "J":
            row[2] = 1
    
    return table

def clarity_continuous(table): # index 3
    for row in table: # (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
        if row[3] == "I1":
            row[3] = 1
        elif row[3] == "SI2":
            row[3] = 2
        elif row[3] == "SI1":
            row[3] = 3
        elif row[3] == "VS2":
            row[3] = 4
        elif row[3] == "VS1":
            row[3] = 5
        elif row[3] == "VVS2":
            row[3] = 6
        elif row[3] == "VVS1":
            row[3] = 7
        elif row[3] == "IF":
            row[3] = 8
    
    return table

def make_continuous(table):
    cut_continuous(table)
    color_continuous(table)
    clarity_continuous(table)
    
    return table

def shrink_tdidt(table):
    new_table = []
    for row in table:
        new_table.append([row[0], row[1], row[2], row[3], row[-1]])
    return new_table

# turns discritized values into strings
def make_string(table):
    new_table = copy.deepcopy(table)
    
    for row in table:
        row[0] = str(row[0])
        row[-1] = str(row[-1])
    
    return table

def get_counts(table):
    my_table = []
    my_table = flatten_table(table)
    values, counts = get_frequencies_string(my_table, -1)
    temp_list = []
    choice = max(counts)
    index = counts.index(choice)
    temp_list.append(values[index])
    temp_list.append(counts[index])
    del values[index]
    del counts[index]
    for i in range(len(values)):
        temp_list.append(values[i])
        temp_list.append(counts[i])
    return temp_list

def flatten_table(table):
    new_table = []
    for my_table in table:
        for row in my_table:
            new_table.append(row)
    return new_table

# used to normalize diamond table
def normalize_table(table): # carat, cut, color, clarity
    carat_list = []  # index 0
    cut_list = []  # index 1
    color_list = []  # index 2
    clarity_list = []  # index 3

    carat_list = get_column(table, 0)
    cut_list = get_column(table, 1)
    color_list = get_column(table, 2)
    clarity_list = get_column(table, 3)

    normalize(carat_list)
    normalize(cut_list)
    normalize(color_list)
    normalize(clarity_list)

    for index in range(len(table)):
        table[index][0] = carat_list[index]
        table[index][1] = cut_list[index]
        table[index][2] = color_list[index]
        table[index][3] = clarity_list[index]

    return table

# get euclidean distance
def distance(v1, v2):
    assert(len(v1) == len(v2))
    dist = math.sqrt(
        sum([(float(v1[i]) - float(v2[i])) ** 2 for i in range(1, len(v1))]))
    return round(dist)

def normalize(xs):
    # Normalize... Use the formula: (x - min(xs)) / ((max(xs) - min(xs)) * .1) DO THIS IN HOLDOUT SO DON'T HAVE TO DO IT IN EVERY DIFFERENT CLASSIFIER FUNCTION??
    for index in range(len(xs)):
        if not ((max(xs) - min(xs)) * .1) == 0:
            xs[index] = int(round((xs[index] - min(xs)) /
                                  ((max(xs) - min(xs)) * .1)))
        else:
            var = 0

    return xs

def get_column(table, column_index):
    column = []
    for row in table:
        if row[column_index] != "NA":
            column.append(float(row[column_index]))

    return column

def get_column_string(table, column_index):
    column = []
    for row in table:
        if row[column_index] != "NA":
            column.append(row[column_index])

    return column

# Standard read csv from python csv library
def read_csv(filename):
    table = []
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            table.append(row)
    return table 

# moves price to the last index... swapping price and "z depth"
def move_price(table):
    
    for row in table:
        temp = copy.deepcopy(row[-1])
        row[-1] = row[6]
        row[6] = temp
    
    return table

# Used to remove the "record" column from the dataset
def remove_record(table):
    new_table = []
    for row in table:
        new_table.append(row[1:11])
    return new_table

# Write csv to table
def write_table(outfile, table):
    with open(outfile, 'w') as writeFile:
        writer = csv.writer(writeFile, delimiter=',')
        writer.writerows(table)
    writeFile.close()

def write_table_header(outfile, table, header):
    with open(outfile, 'w') as writeFile:
        fieldnames = header
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames)
        writer.writeheader()
        writer = csv.writer(writeFile, delimiter=',')
        writer.writerows(table)
    writeFile.close()

def get_frequencies(table, column_index):
    column = sorted(get_column(table, column_index))
    values = []
    counts = []

    for value in column:
        if value not in values:
            values.append(value)
            # first time we have seen this value
            counts.append(1)
        else:  # we've seen it before, the list is sorted...
            counts[-1] += 1
    return values, counts

def get_frequencies_string(table, column_index):
    column = sorted(get_column_string(table, column_index))
    values = []
    counts = []

    for value in column:
        if value not in values:
            values.append(value)
            # first time we have seen this value
            counts.append(1)
        else:  # we've seen it before, the list is sorted...
            counts[-1] += 1
    return values, counts
    

def compute_bin_frequencies(values, cutoffs):
    freqs = [0] * len(cutoffs)
    for val in values:
        for i, cutoff in enumerate(cutoffs):
            if val <= cutoff:
                freqs[i] += 1
                break
    return freqs

def compute_equal_widths_cutoffs(values, num_bins):
    # first things first...need to compute the width using the range
    values_range = max(values) - min(values)
    print("max(values): ", max(values))
    width = values_range / num_bins
    # width is a float...
    # using range() we can compute the cutoffs
    # if possible, your application allows for it, convert min, max, width
    # to intgers
    # we will work with floats 
    # np.arange() works with floats
    cutoffs = list(np.arange(min(values) + width, max(values) + width, width))
    # round each cutoff to 1 decimal place before we return it
    cutoffs = [float(round(cutoff, 1)) for cutoff in cutoffs]
    return cutoffs

def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

'Tensorflow Functions'
# 'carat', 'cut', 'color', 'clarity', 'price'
CSV_COLUMN_NAMES = ['carat', 'cut',
                    'color', 'clarity', 'Price']
SPECIES = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10']

def maybe_download():
    train_path =  "diamond_training.csv"
    test_path = "diamond_test.csv"

    return train_path, test_path

def load_data(y_name='Price'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download() # "diamond_training.csv", "diamond_test.csv"

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.io.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Price')

    return features, label