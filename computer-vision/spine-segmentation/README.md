# Vertebra segmentation from a frontal spine X-ray

**Problem.** Split a frontal full-spine X-ray into one image per vertebra, labelled from the sacrum up to C1.

**Method.** Classical image processing, no learned model. Part 1 enhances contrast with gamma correction and CLAHE, then adds Sobel and Canny edges. Part 2 cuts the image into 25 horizontal strips and keeps the brightest region in each strip to isolate the spine column. Part 3 sums pixel values along each row, smooths the profile with a moving average, finds local minima as the gaps between vertebrae, and crops between them.

**Results.** The notebook shows only qualitative crops. There is no quantitative evaluation, and the cervical region is hard to separate in a frontal view because the vertebrae overlap.

**Data.** Not included. The notebook expects two X-ray images (`4.jpeg` and `Spine_Segments.jpeg`) in `/content/`. Image outputs were removed from this copy of the notebook.

**Note.** This was a group term project with 3 members.

**Run.** `pip install -r requirements.txt`, place your own image at the paths above, and run the cells in order.
