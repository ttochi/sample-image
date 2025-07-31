FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.6

RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision

COPY sample.py .
