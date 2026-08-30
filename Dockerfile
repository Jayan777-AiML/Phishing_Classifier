FROM python:3.10-slim 
# python:slim-buster is a specific, official Docker base image tag 
# used to deploy Python applications in lightweight container environments

WORKDIR /app

COPY . /app 

COPY venv phishfiner_venv

RUN pip install -r requirements.txt 

RUN pip install "Cython<3" "setuptools<70"
RUN pip install --no-build-isolation evidently==0.1.58.dev0

EXPOSE 5010

CMD ["python3", "app.py"] 
#The first value (python3) is the executable entry point — the program that will be run inside the container.,,  The second value (app.py) is the script file that will be passed as an argument to the executable. 
