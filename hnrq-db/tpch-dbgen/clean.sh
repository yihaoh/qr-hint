#!/bin/bash

# Loop through all files with .tbl extension in the current directory
for file in *.tbl; do
    # Check if the file exists to avoid errors if no files match the pattern
    if [[ -f "$file" ]]; then
        echo "Processing $file ..."
        # Use sed to remove the last character "|" in each line and overwrite the file
        sed -i 's/|$//' "$file"
    fi
done

