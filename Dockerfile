FROM kpavlovsky/python3.7
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -U PaddleDesktop-EasyEdge-linux-m1-r1-x86/python/BaiduAI_EasyEdge_SDK-1.3.7-cp37-cp37m-linux_x86_64.whl
RUN apt-get update && apt-get -y upgrade && apt-get -y install git libgl1
EXPOSE 20241
CMD ["python"]
