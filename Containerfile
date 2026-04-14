FROM registry.redhat.io/ubi10/python-312-minimal:latest

# Listen on port 8501
EXPOSE 8501/tcp

# Set the working directory in the container
WORKDIR /projects

# Copy the content of the local src directory to the working directory
COPY . .

# Install any dependencies
RUN \
  if [ -f requirements.txt ]; \
    then pip install -r requirements.txt; \
  elif [ `ls -1q *.txt | wc -l` == 1 ]; \
    then pip install -r *.txt; \
  fi

# Specify the command to run on container start
ENTRYPOINT ["python", "-m", "streamlit", "run", "main_streamlit.py"]
