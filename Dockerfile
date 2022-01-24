FROM python:3.7

LABEL version="0.1.0"
LABEL maintainer="Yuncheng Wu"

COPY . /splitnndemo
WORKDIR /splitnndemo

# Setup environment
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install jupyterlab

# Expose port for jupyter lab (remove if not using jupyter on host)
EXPOSE 8888

# Enter into jupyter lab (remove if not using jupyter on host)
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
