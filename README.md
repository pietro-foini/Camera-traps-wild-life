# ML-Piane-di-Sopra

## How to install

### Local virtual environment

We suggest to use [PyCharm Community](https://www.jetbrains.com/pycharm/download/#section=windows) for following 
steps 2-8.

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

Now you're all set! 🎉 Happy coding! 😄✨

### Using Docker

1. Install Docker: Visit the official Docker website (https://www.docker.com/) and follow the installation instructions 
for your operating system; 
2. Clone the repository;
3. Build the Docker image: navigate to the project's root directory and run the following command to build the Docker image:
   1. `docker build -t project_name .`
4. Run the Docker container: Once the image is built, start a container with the following command:
   1. `docker run -it project_name`

🚀 This will launch the project within the Docker container! 🐳

-----

## Model

Currently, the model being used is `EfficientNetB0` (https://keras.io/api/applications/), which
was implemented to undergo *fine-tuning* using the custom dataset. The choice of this model was
driven by its high accuracy and relatively low number of parameters. More recent series of the same
model result in a decrease in computational performance.

The training sessions were conducted using an NVIDIA GPU GeForce 940MX.

## Dataset

The dataset used for training is available at the Google Drive [link](https://drive.google.com/file/d/1DebJb2638-DqQDnvEwk7CoMHNx1Ipf03/view?usp=drive_link) (~ 2.5 GB).

The current dataset has been obtained by combining multiple sources of data available online in order to assemble a
dataset of images captured by camera traps in both daytime and nighttime settings.
The currently available image classes are as follows:

| label             | setting     | count |
|-------------------|-------------|-------|
| None_of_the_above | day         | 3000  |
| None_of_the_above | night       | 400   |
| badger            | day         | 955   |
| badger            | night       | 1474  |
| badger            | unspecified | 18    |
| bear              | day         | 985   |
| bear              | night       | 420   |
| bear              | unspecified | 779   |
| bird              | unspecified | 2777  |
| boar              | day         | 1287  |
| boar              | night       | 675   |
| boar              | unspecified | 775   |
| cat               | day         | 1045  |
| cat               | night       | 935   |
| cat               | unspecified | 4759  |
| chicken           | unspecified | 680   |
| cow               | day         | 1351  |
| cow               | night       | 103   |
| cow               | unspecified | 1138  |
| deer              | day         | 3805  |
| deer              | night       | 2286  |
| deer              | unspecified | 561   |
| dog               | day         | 1360  |
| dog               | night       | 124   |
| dog               | unspecified | 3291  |
| fox               | day         | 1408  |
| fox               | night       | 1320  |
| fox               | unspecified | 8     |
| hare              | day         | 20    |
| hare              | night       | 1262  |
| hare              | unspecified | 5110  |
| horse             | unspecified | 62    |
| human             | unspecified | 2980  |
| squirrel          | unspecified | 2775  |
| vehicle           | unspecified | 2829  |
| weasel            | day         | 1907  |
| weasel            | night       | 1119  |

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