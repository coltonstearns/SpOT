#
# Download preprocessed datasets into the current folder.
#

# Preprocessed nuScenes datatset (28GB compressed)
wget http://download.cs.stanford.edu/orion/spot/nuscenes-centerpoint-preprocessed.zip
unzip nuscenes-centerpoint-preprocessed.zip
rm nuscenes-centerpoint-preprocessed.zip

# Preprocessed Waymo Open dataset for evaluation (20GB compressed)
wget http://download.cs.stanford.edu/orion/spot/waymo-centerpoint-preprocessed.zip
unzip waymo-centerpoint-preprocessed.zip
rm waymo-centerpoint-preprocessed.zip

# Preprocessed Waymo open dataset for training
wget http://download.cs.stanford.edu/orion/spot/waymo-centerpoint-cthresh0.6-preprocessed.zip
unzip waymo-centerpoint-cthresh0.6-preprocessed.zip
rm waymo-centerpoint-cthresh0.6-preprocessed.zip