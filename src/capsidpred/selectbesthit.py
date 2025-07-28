import csv
import collections
import sys
import os

def select_max_bitscore(input_file):
    """
    Selects rows with the highest bitscore for each unique value in the first column and overwrites the input file.
    :param input_file: input file path
    """
    # Define result named tuple
    Result = collections.namedtuple("Result", ["phrog", "db", "protein", "bitscore"])

    # Dictionary to store the best result for each protein
    best_results = {}

    # Read input file and find the highest bitscore for each protein
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        for row in reader:
            phrog, db, protein, bitscore = row[0], row[1], row[2], float(row[3])
            current_best = best_results.get(phrog)
            if current_best is None or bitscore > current_best.bitscore:
                # Update the best result for the protein
                best_results[phrog] = Result(phrog, db, protein, bitscore)

    # Write best results to a temporary output file
    temp_output_file = input_file + '.tmp'
    with open(temp_output_file, 'w') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for result in best_results.values():
            writer.writerow([result.phrog, result.db, result.protein, result.bitscore])
    
    # Replace the original file with the temporary file
    os.replace(temp_output_file, input_file)

# Main function to handle command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python select_max_bitscore.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    select_max_bitscore(input_file)
    print(f"Processed {input_file} and selected rows with the highest bitscore for each unique value in the first column.")