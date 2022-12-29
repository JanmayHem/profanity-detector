# Profanity Detector
A python based Machine Learning model, which detects profane language and returns true or false with respect to a social media environment (yeah, have to consider things like free speech shm).  

<!-- This is under development. [================>. . . .] -->

## How to use
1) Download all .py files or [clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
2) Download all files in the Input Files and arrange them [accordingly](https://github.com/JanmayHem/profanity-detector/blob/main/Input%20Files/README.md).
<br>&emsp;• Alternatively, you can access the same input files from [Kaggle](https://www.kaggle.com/code/psvenom/balls/data).
3) Run the \_profanity_.py file. 
<br>&emsp;• This will read all the data, preprocess it, form a proper format, and feed the model. 
<br>&emsp;• Then the model is saved.
4) Run the data-csv-generator.py files.
<br>&emsp;• This makes the csv file which we will use to compare with the inputs.
5) The main file is where the function is, which takes in the Statement in String form and returns True or false based on the profanity score given by the model trained.

## Details 
Keras, which is an interface of TensorFlow library, has been used for the training of this model. It has 4 Epochs, each of which take about 40 mins max. This model was designed to suit the likes of a social media app, hence using profane words with \*,@,#, etc will be allowed as people do expect to express their views and not hurt public sentiment using such tricks. Note that the sample used in the \_profanity_.py file, and some Input Files contains the use of strong language. 

## Why this project?
Mainly to assist the app mentioned above. But also to explore and get into the world of Machine Learning! Do let me know if there are some inconsistencies. 

## Contributing
• Pull requests are welcome, why else we on GitHub for eh. 

• For major changes, please open an issue first
to discuss what you would like to change. Please make sure to update tests as appropriate. 

• Aiming to make this a proper Python Library on PyPI.

• Open to discussions as well! 

## License

This repository comes under the [MIT License](https://choosealicense.com/licenses/mit/). Kindly check all the details.

-- <br>
Made with :heart: by [@JanmayHem](https://github.com/JanmayHem) 
<br><br>[![Tech Stack Used](https://skillicons.dev/icons?i=py,tensorflow,git,github)](https://skillicons.dev)
