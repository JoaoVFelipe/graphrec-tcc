FROM python:3.6.13
RUN apt-get update

# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /
#to COPY the remote file at working directory in container
COPY . ./

RUN pip install -r ./requirements.txt
# Now the structure looks like this '/usr/app/src/test.py'
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.
# CMD [ "python", "-u", "./citeulike-graphrec.py"]
CMD [ "python", "-u", "./citeulike-graphrec-second_version.py"]
