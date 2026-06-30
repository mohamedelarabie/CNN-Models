SetUp For Linux 

```bash 
conda create -n  cnn python=3.10
conda activate cnn
conda install pip 
pip install -r requirements.txt
```

SetUp For Windows

``` bash
winget install Python.Python.3.10
py -3.10 -m venv cnn
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser (Option)
cnn\Scripts\activate
pip install -r requirements.txt
```