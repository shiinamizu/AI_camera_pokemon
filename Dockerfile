FROM python:3.14
# コンテナ内の作業ディレクトリを作成
WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r ./requirements.txt
