#!/bin/bash

# Assuming your subdirectories are in the current directory
# You may need to adjust the path accordingly if they are in a different location

parent_directory="/home/joshm/school/CS5510-Robit/midterm/CS-5-6510-midterm/Problem8/data/full_store/train"  # Replace with the actual path to your parent directory

# List of directories to exclude
exclude_dirs=("bakery" "bookstore" "clothingstore" "deli" "florist" "grocerystore" "jewelleryshop")

# Loop through the subdirectories
for dirname in "$parent_directory"/*/; do
    # Get the name of the subdirectory without the path
    dirname="$(basename "$dirname")"
    
    # Check if the dirname is in the exclude_dirs list
    if [[ " ${exclude_dirs[@]} " =~ " $dirname " ]]; then
        echo "Skipping $dirname"
        continue  # Skip this directory
    fi
    # Change to the subdirectory
    cd "$parent_directory/$dirname" || exit

    # Create the 'valid' directory
    mkdir -p "../../valid/$dirname"
    
    
    # Count the number of files or directories in the current subdirectory
    num_items=$(ls -1U | wc -l)
    
    # Calculate 20% of the total number of items
    twenty_percent=$((num_items * 20 / 100))
    
    # Move the first 20% of items to the 'valid' directory
    ls | head -n "$twenty_percent" | xargs -I {} mv {} "../../valid/$dirname/"
    
    # Return to the parent directory
    cd "$parent_directory" || exit
done
