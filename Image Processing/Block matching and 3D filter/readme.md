## Block Matching and 3D Filtering
### Colllabortaive filtering process.
### It is Grouping based on the similarity of the pixels that are extracted from the image.

### A block is a grouped if its dissimilarity with a reference fragment is less than a threshold.

### All blocks in group are then  stacked together to form a 3D cylinder-like shapes.

### Filtering is done on every block group. Linear transform is applied to the 3D block group then transform is inverted to get the filtered block group. 

### Image transformed back to its 2D form.