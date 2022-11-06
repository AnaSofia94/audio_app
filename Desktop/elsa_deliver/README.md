# Audio inspection




# Pip installation 
Create a python 3 virtual environement and activate it 

$ virtualenv -p python3 env

$ source env/bin/activate

Internal documentation: 
    http://127.0.0.1:8000/redoc 
    
#Audio inspection

- @app.get('/home', response_class=HTMLResponse) endpoint:
  . returns the home page
  
- @app.post("/submitform") endpoint: 
  . returns the result of the analysis in a table form

more information can be found here: 127.0.0.1:800/docs

External documentation: external documentation.pdf


To run Docker: 
  -  docker build -t audio_info . 
  -  docker run -p 80:80 audio_info


   
