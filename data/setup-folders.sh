#!/bin/bash

#!/bin/bash

# List of folders to recreate
folders=(
  "dynamics-files"
  "work-moments-files"
  "wcf-files"
  "wpd-files"
  "photon-moments-files"
  "pcf-files"
  "ppd-files"
)

# Loop through the list and recreate each folder
for folder in "${folders[@]}"; do
    if [ -d "$folder" ]; then
        # echo "Removing existing folder: $folder"
        rm -rf "$folder"
    fi
    # echo "Creating folder: $folder"
    mkdir -p "$folder"
done

echo "Folders reset/created successfully."