1. Create virtual enviroment in a Linux Terminal
```
   mkvirtualenv <env_name>
```   

2. Activate virtual enviroment
```
   workon <env_name>
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
```

7. Desactivate deactivate
```
   <env_name> $ deactivate
```
