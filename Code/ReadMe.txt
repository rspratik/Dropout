Instructions
------------
Dropout study is carried out on two datasets. MNIST image dataset and Newsgroup text dataset.
Necessary comments has been provided in each of the project files.
Each of following files has tensorflow code with slight variation. They can be executed independently. 
They will create respective log files which can be rendered over the tensorboard.

Note: 
1. I observed that newsgroup experiment files(python) gave runtime issues on windows machine but ran abolutely fine on linux instance (I have attached the respective text files having output).
2. Due to limited computation, current value of epoch is set to 50. For better and expected results, higher value of epoch is recommended.


1.MNIST (Image dataset)
-----------------------------------------------------------------------------------
mlp_standard.py 
mlp_standard_decay.py
mlp_dropout.py
mlp_dropout_nested.py
mlp_dropout_nested_adams.py
mlp_dropout_nested_decay.py


2.Newsgroup (Text/corpus dataset) with 5 out of 22 categories used for experiment.
-----------------------------------------------------------------------------------
t_mlp_standard.py -- 86.8
t_mlp_dropout.py  -- 84.12
t_mlp_dropout_nested.py --84.40

-----------------------------------------------------------------------------------