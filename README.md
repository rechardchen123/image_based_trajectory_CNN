# image_based_trajectory_CNN
This is a convolutional neural network projet about trajectory image process for maritime situational awareness.

## process AIS data
This is the first part of the whole project. The raw AIS data files use the .csv format to record the vessel movement trajectories and motion information. The numerical fileds of AIS in our project are inclued: MMSI (Maritime Mobile Service Identify), Latitude, Longitude, Speed, Heading, Timestamp. The project uses these field to build the vessel movement trajectories and use the convolutional neural network to classify the vessel movement patterns (static, curise, manoeuvering).
The process AIS data contains these steps:
### read and process data
1. read the file from local repository.
2. transfer the timestamp to local time format
3. transfer the longitude and latitude to the regular format to read and recognise
4. transfer the heading and speed
5. get a range of data. lontitude(116-119) and latitude(36-39)
6. delete unnecessary field and store them.
The total dataset can be downloaded from the link:
[AIS data raw files and after processed file](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/Ek_9b7cAKuBGto3lhzsEXpEBE_8uWfSw9wNgl1_AopJRlA?e=6CbROA).
The merged_data.csv is the processed AIS data.
### clean and group the data by mmsi
After getting the processed AIS data, I use the pandas dataframe to read the csv file. I use the MMSI unique attribute to group the data by MMSI. AFfter processed data is here: [gropud by MMSI](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/ErDiKs_UPLdHrD4R60aNpTsB8p35oitzbV_svfSKwC0XUA?e=xIIXBn).

### delete small csv file
After spliting, we get the MMSI files and find that there produced a lot of small files. It cannot help to build the trajectories. So, the delete small csv file is to delete them. 

### groupby mmsi by day 
When splited the AIS data file by mmsi, the different days for the same MMSI cannot split and the 'grouby mmsi by day' is used to realize the function.
After that, the 'process_ais_data' can be finished.
The data file can be found here: [process_ais_data_by_mmis_day](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/EpzOkul_kEtNvw_fHs3r2TgBKDgDdaqJ1lgktHXsIHBWQw?e=JFLTmY).

## delta_time_delta_speed_split_per_day
This function is used to calculate time difference and speed difference for the successive two AIS points. After that the data fields is: 'MMSI', 'Longitude', 'Latitude', 'Speed', 'Heading', 'Day', 'time_to_seconds', 'delta_time', 'delta_speed', 'delta_heading'. 
The data file is: [delta_time_delta_speed_split_per_day](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/EuPC6slnm9BJgtmxgL_zqgsBPygAB87FPJMrD9SEgomvZg?e=EOIhRQ).

## split abnormal trajectory per day
The 'split_abnorma_ais_per_day' is used to split the AIS data per day. Due to the AIS devices uneven sampling phenomena, using this function can split the abnormal trajectory. Before doing this, I calculate the 'delta_heading'. If the delta_heading is abnormal, it should be split from the same day AIS trajectory file. Here, the threshold value of delta_heading is 20. It refers to the navigational maneuvring knowledge.
The data file is: [split abnormal trajectory per day](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/EkPOto5KHglLnsEvpXLZN6sB7rKr2rb5yZGYm_3AO-B-Og?e=oVhbCM).

## trajectory compensation
This function is used to compensate some missing AIS points due to the AIS instability.

## trajectory generation
In this function, we three steps for generation trajectories that they contain the
motion characters. 
First, generate the trajectory pictures.
Second, determine the target areas.
Third, determine the number and value of pixels of the image.
And then, output the images and generate the arraies for classifying.
the data file is: [trajectory generation](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/ElSN7xSz1kJAr4NscJegkeoB221ZfwC5kTyfsQ8sBTmsQw?e=UvGhxG).

## trajectory generation after clip
In order to get a uniform size image for the CNN training, I use the 'clip_image' function to clip the image into one size (width:360, height:490). The image file after clip: [clip image](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucesxc0_ucl_ac_uk/EglrQ8Hq7FhMuuB-uIHPuMQBlXYAIJTbGzB2qldeq7N6tw?e=H8xtbC).

## training and tesing processing for trajectory classification in tensorflow
It contains two folders. One is just using cpu to training and the other is used the GPU telsa p100 to train the model.

### data_process
In this function, it should transfer the raw image into 'tfrecord' format to accelerate the reading speed for TensorFlow. And also, labelling all the image files to help calculate the accuracy. 

### build_network
This function is used to build the convolutional neural network. It contains convolution layers, pooling layers, fully connected layers, softmax layer. And also, the loss function and evalutaion function are contained. 

## training_network
This function contains read and decode the 'tfrecord' and define the placeholder for the CNN. And the training data feeding flow is used to get the training result. 

If you have some confusion or troubles, please feel freely to contact with me. 
email: <xiang.chen.17@ucl.ac.uk>
github: <https://github.com/rechardchen123/image_based_trajectory_CNN>
twitter:<https://twitter.com/tony_chen123?lang=zh-cn>





