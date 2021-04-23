# pipeline_cv_imagewoof
Pipeline means that you can both train the model on your own parameters, 

and predict classes from the folder for unmarked images

You can get the training data here

https://github.com/fastai/imagenette

## Allowed arguments

![](https://github.com/Windmen05/pipeline_cv_imagewoof/blob/master/data%20for%20README/argparser_help.png "Arguments")


## How to use it?

### Predict

For the prediction, you just need to specify the path to the folder using -fp,

the pipeline will take the model I trained, which showed a quality of 75% during validation,

or you can specify the name of the model that should be in the data / folder, also after the prediction,

the model will write the result to a json file, the name for the file you can specify using -on

![](https://github.com/Windmen05/pipeline_cv_imagewoof/blob/master/data%20for%20README/fast_start.png "Fast start")

### Train

For training, you are invited to take my version of the neural network architecture and train it on your hyperparameters,

after training, the model will be saved in the folder data/

also in data/ you can find the models that showed the best val_loss

![](https://github.com/Windmen05/pipeline_cv_imagewoof/blob/master/data%20for%20README/how_to_train.png "How to train")
