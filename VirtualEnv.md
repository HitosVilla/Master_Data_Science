1. Create virtual enviroment in a Linux Terminal
```
   python3 -m venv <env_name>
```   

2. Activate virtual enviroment
```
   source <env_name>/bin/activate
```   

3. Install a Kernell
```
	<env_name> $ pip install ipykernel
	<env_name> $ ipython kernel install --user --name=<env_name>
```	

4. Install required libraries
```
   <env_name> $ pip install ......
   <env_name> $ ....
```

5. Run your notebook in the new kernell to test everything works fine.

6. Export requirements file with the list of installed libraries
```
   <env_name> $ pip freeze > requirements.txt
   pip install -r requirements.txt
```

7. Desactivate 
```
   <env_name> $ deactivate
```
8. Remove 
```
source venv/bin/activate
pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
deactivate
rm -r venv/
```
