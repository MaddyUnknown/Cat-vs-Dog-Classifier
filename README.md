# Cat-vs-Dog-Classifier
A binary classifier to check whether an image contains a cat or a dog

## How the classifier works
`predictor.py` accept the following command line argument:
        - "--" before the name indicates a folder contain the images to evaluate, i.e. `--folder_name`
        - "-"  before the name indicates a txt file containing the location of the images to evaluate, i.e. `-sample.txt`
        - an argument without prefix means an single image path `-file_loc`

## Repo Contents:
- `Models` contain all the trained models and there accuracy
- `Previous_versions` contain the previously structured ipnb file for models.
- `Test_Image` contain image to demonstrate how predictor works
- `Augmentation_script.ipynb` augments the original dataset and save the augmented dataset
- `Classifier_pytorch_v3.ipynb` contain the current train loop and structure of network
- `predictor.py` is the driver program that applies the classifier to given images

## Sample Images:

![application example](https://github.com/MaddyUnknown/Application-form-Tkinter/blob/master/Readme_img/Application%20form.png)
> ###### GUI of Application

## Languages and Tools used:
  ### Language:
  `Python 3.7.3`
  ### Modules
  `PyTorch 1.0.0`
  `Pandas 0.25.3`
  `Matplotlib 2.1.1`
  `Numpy 1.19.2`
  `PIL 8.1.2`
