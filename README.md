# Road Network Pixelation: A Missing Traffic Flow Imputation Method Inspired by Video Inpainting Techniques


## Dataset
Due to the large amount of private data used in this study, we have only uploaded the code. Some sample data used in this study are shown in the below table.

| plate number | pass time | checkpoint | lat | lng | plate color | speed | ... |
| :------: | :------: | :------: |  :------: | :------: | :------: | :------: | :------: |
|*AD**39|00:02:17|A156**|29.675166|121.581432|blue|32| ... |
|*J2**29|00:02:18|B326**|29.558313|121.583809|blue|42| ... |
|*A4**38|00:15:10|A245**|29.679775|121.450264|blue|-| ... |
|*Z8**6W|05:13:15|D753**|29.577226|121.509046|blue|54| ... |
|*A0**29|12:43:54|F142**|29.655941|121.472525|yellow|36| ... |
|*AC**98|16:22:28|B674**|29.655643|121.413580|blue|-| ... |
|*HU**11|13:43:13|ZD30**|31.452642|122.205742|yellow|43| ... |
|*NL**63|17:53:29|X170**|29.614257|119.524781|blue|53| ... |
|*AT**39|19:12:03|TE09**|31.842688|121.325379|blue|-| ... |
|...|...|...|...|...|...|...|...|

To meet the privacy policy, we encrypted the license plate number and the checkpoint number in the sample, and the latitude and longitude are offset nonlinearly. The data are stored in a total of 78 tables, and each table stores one day of data.

## Usage
Step1: Use the road network pixelation algorithm in Section IV of this paper to convert the data into traffic flow images.  
Step2: The data needs to be convert into the following form.

* mask: shape=(64, 64) type=numpy.float64
* data:
  * key: yyyy-MM-dd HH:mm:ss type:datetime
  * value:
    * real: shape=(64, 64) type=numpy.float64
    * loss: shape=(64, 64) type=numpy.float64
    * loss_mask: shape=(64, 64) type=numpy.float64
    * loss_indices: len=(checkpoint_count * loss_ratio) type=list[(int, int)]


## Requirements
* Python 3
* PyTorch
* NumPy
* Pandas
* scikit-learn
* tqdm
* CUDA
* CUDNN

## Run
```
python main.py --parameters xx
```

## Parameters
* batch_size
* epochs
* loss_ratio
* random_seed
* save_path
* learning_rate
* mask_path
* dataset_file_path
* rnn_type