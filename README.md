# ML-Piane-di-Sopra

## How to install

### Local virtual environment

We suggest to use [PyCharm Community](https://www.jetbrains.com/pycharm/download/#section=windows) for following 
steps 2-7.

1. Install Python 3.9: Make sure you have Python installed on your system. You can download it from the official Python 
website (https://www.python.org/) and follow the installation instructions for your operating system;
2. Clone the repository;
3. Create a virtual environment;
4. Activate the virtual environment;
5. Mark `src` folder as root directory;
6. Install project dependencies: 
   1. `pip install -r requirements.txt`
7. Run the commands for further project dependencies: 
   1. `poetry lock --no-update`
   2. `poetry install`
8. Run main project script:
   1. `python src/main.py`

### Using Docker

1. Install Docker: Visit the official Docker website (https://www.docker.com/) and follow the installation instructions 
for your operating system; 
2. Clone the repository;
3. Build the Docker image: navigate to the project's root directory and run the following command to build the Docker image:
   1. `docker build -t project_name .`
4. Run the Docker container: Once the image is built, start a container with the following command:
   1. `docker run -it project_name`

This will launch the project within the Docker container.

-----

## Model

Currently, the model being used is `EfficientNetB0` (https://keras.io/api/applications/), which
was implemented to undergo *fine-tuning* using the custom dataset. The choice of this model was
driven by its high accuracy and relatively low number of parameters. More recent series of the same
model result in a decrease in computational performance.

The training sessions were conducted using [Google Colab](https://research.google.com/colaboratory/).

## Dataset

The dataset used for training is available at the Google Drive link (~ 2.5 GB).

The current dataset has been obtained by combining multiple sources of data available online in order to assemble a
dataset of images captured by camera traps in both daytime and nighttime settings.
The currently available image classes are as follows:

| label             | setting     | count |
|-------------------|-------------|-------|
| None_of_the_above | day         | 3000  |
| None_of_the_above | night       | 400   |
| badger            | day         | 955   |
| badger            | night       | 1491  |
| badger            | unspecified | 22    |
| bear              | day         | 991   |
| bear              | night       | 435   |
| bear              | unspecified | 959   |
| bird              | unspecified | 2809  |
| boar              | day         | 1287  |
| boar              | night       | 691   |
| boar              | unspecified | 830   |
| cat               | day         | 1045  |
| cat               | night       | 935   |
| cat               | unspecified | 4765  |
| chicken           | unspecified | 681   |
| cow               | day         | 1364  |
| cow               | night       | 111   |
| cow               | unspecified | 1275  |
| deer              | day         | 3806  |
| deer              | night       | 2296  |
| deer              | unspecified | 562   |
| dog               | day         | 1360  |
| dog               | night       | 124   |
| dog               | unspecified | 3315  |
| fox               | day         | 1414  |
| fox               | night       | 1335  |
| fox               | unspecified | 8     |
| hare              | day         | 20    |
| hare              | night       | 1262  |
| hare              | unspecified | 5163  |
| horse             | unspecified | 63    |
| human             | unspecified | 2980  |
| squirrel          | unspecified | 2813  |
| vehicle           | unspecified | 3107  |
| weasel            | day         | 1906  |
| weasel            | night       | 1120  |

The dataset folder structure is then organized as follows:

    root
     └── dataset
          ├── label1
          │   ├─ image1.jpg
          │   ├─ image2.png                              
          │   └─ ...
          └── label2
              └── ...

The filename of each image is defined as follows:

    {referenceNameDataset}_{nameLabel}_{timeCondition}_{progressiveIndex}.jpg

N.B. The underscores are only used as separators, otherwise *CamelCase* notation has been used.

The `timeCondition` field can be: 'day', 'night', 'unspecified'.

Useful dataset links:

- NTLNP (wildlife image dataset): https://paperswithcode.com/dataset/ntlnp-wildlife-image-dataset

- CCT20 (subset): https://lila.science/datasets/caltech-camera-traps

- Sheffield: https://figshare.shef.ac.uk/articles/dataset/Badger_datasets_for_image_recognition/8182370/1

- ENA24: https://lila.science/datasets/ena24detection

- LilaMissouri: https://lila.science/datasets/missouricameratraps

- WCS: https://lila.science/datasets/wcscameratraps

- PennFudan: https://www.cis.upenn.edu/~jshi/ped_html/

## License

MIT

## Contacts

Please open an issue or contact pietro.foini1@gmail.com with any questions.