#!/bin/bash

#this script counts the number of images for each genus and species in a given folder structure and writes the counts to a text file
#it also renames any .jpeg files to .jpg to standardize file extensions


image_base_path="/work/soghigian_lab/abdullah.zubair/rerun4/ImageBase"

#set output file path
output_file="image_counts.txt"


declare -A genus_counts
declare -A species_counts

#rename .jpeg files to .jpg
find "$image_base_path" -type f -name "*.jpeg" -exec bash -c 'mv "$0" "${0%.jpeg}.jpg"' {} \;


for subfolder in "$image_base_path"/*; do
  if [ -d "$subfolder" ]; then
    genus_species=$(basename "$subfolder")
    genus=$(echo "$genus_species" | cut -d '_' -f 1)
    species=$(echo "$genus_species" | cut -d '_' -f 2)

    image_count=$(find "$subfolder" -type f -name "*.jpg" -o -name "*.png" | wc -l)


    genus_counts["$genus"]=$((genus_counts["$genus"] + image_count))
    species_counts["$genus_species"]=$image_count
  fi
done


echo "Genus Counts:" > "$output_file"
for genus in "${!genus_counts[@]}"; do
  count=${genus_counts["$genus"]}
  echo "$genus: $count" >> "$output_file"
done

echo -e "\n\n\n" >> "$output_file"


echo "Species Counts:" >> "$output_file"
for species in "${!species_counts[@]}"; do
  count=${species_counts["$species"]}
  echo "$species: $count" >> "$output_file"
done
