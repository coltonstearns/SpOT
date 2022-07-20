The preprocessed nuScenes and Waymo Open data are available for download. Note that this data can be quite large depending on the category, so please edit [`download_data.sh`](download_data.sh) to only download and unzip what you need. To download all datasets run:
* `bash download_data.sh`

If preferred, the nuScenes dataset is also available to download manually from Google drive: [nuScenes (28GB)](https://drive.google.com/file/d/1eTDCrjxg8y2v5-2N-chKrrkpu-Zb-n3T/view?usp=sharing).

Configurations for each dataset that are passed into the training/testing scripts are located in [`configs`](configs). By default, these should not need to be changed.
