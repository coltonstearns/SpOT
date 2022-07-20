We use the visualizer from SimpleTrack. To get Waymo visualizations, do the following:

(1) Run SimpleTrack Waymo preprocessing to get appropriate preprocessed data folder.
(2) Run conversion_bin2npz.py to get a preprocessed folder of the predicted ".bin" tracking file.
(3) Run visualize.py with the Waymo preprocessing folder, the GT preprocessed folder, and the tracking folder.