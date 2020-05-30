FROM centos:latest

RUN yum install python3* -y
	
RUN pip install pandas

RUN  pip install numpy

RUN  pip install sklearn 

RUN  pip install pillow

RUN  pip install scipy

RUN  pip install matplotlib

RUN  pip install pandas

RUN  pip install tensorflow

RUN  pip install keras

RUN  pip install opencv-python

RUN  pip install seaborn

RUN  pip install scikit-learn

EXPOSE 22

ENTRYPOINT [ "python3" ] 

CMD [ "-h" ]
