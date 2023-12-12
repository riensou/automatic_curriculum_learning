1. Create a conda environment that will contain python 3:
```
conda create -n acl python=3.9
```

2. activate the environment (do this every time you open a new terminal and want to run code):
```
source activate acl
```

3. Install the requirements into this conda environment
```
cd src
pip install -r requirements.txt
```

4. Allow your code to be able to see 'acl'
```
$ pip install -e .
```