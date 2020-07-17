#!/bin/python

import os
import csv
import argparse
from shutil import copy2

def opts_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--dir', type=str, required=True)
    return parser

def main():
    # Parse options
    parser = opts_parser()
    args = parser.parse_args()

    # Open dataset file
    dataset = open(args.input, newline='')
    reader = csv.reader(dataset)
    data = list(reader)

    # Variables for progress counter
    lines = len(data)
    i = 0

    # Analyze data in dataset
    for row in data:
        # Assign image name and state to variables
        image = args.dir + '/' + row[1] + '/' + row[0] + '.png'
        state = row[2]

        # Print image information
        print('({}/{}) Processing image: {}'.format(i + 1, lines, image))
        i += 1

        # Determine action to perform
        if state == '0':
            # Attempt to move the file
            try:
                copy2(image, '../data/no_bird/' + row[0] + '.png')
                print('- Copy to no_bird')
            except FileNotFoundError:
                print(' - Failed to find file')
        else:  # Attempt to move the file
            try:
                copy2(image, '../data/bird/' + row[0] + '.png')
                print(' - Copy to bird')
            except FileNotFoundError:
                print(' - Failed to find file')

if __name__ == '__main__':
    main()
