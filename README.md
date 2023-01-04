"# bollon-detection" 

# Recomendation

## First time configuration

[Create a project environment](https://code.visualstudio.com/docs/python/tutorial-flask#_create-a-project-environment-for-the-flask-tutorial)

If you get an error like - Activate.ps1 cannot be    
loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at         
https:/go.microsoft.com/fwlink/?LinkID=135170. Try this - 
```
Set-ExecutionPolicy Unrestricted -Scope Process
```

### Recommending using python 3.8 for all libraries' compatibility
#### 1. Install virtualenv library
```
sudo pip3 install virtualenv
```
#### 2. Setup a virtual enviornment named 'env'
```
python3.8 -m venv env
```
#### 3. Activate 'env'
```
source env/bin/activate
```

## Installing Requirements

``` 
pip install -r requirements.txt
```
or
```
pip3 install -r requirements.txt
```

## Starting the app

```
python app.py
```
## Exiting the app

```
Ctr + c
```