
```markdown
# YOLOv8 Keypoint Detected

## Instalação de Dependências

Para começar, você precisará instalar as dependências necessárias. Certifique-se de ter o Python e o pip instalados em seu ambiente. Em seguida, execute os seguintes comandos:

```bash
pip install mediapipe
pip install opencv-python
```

## Criação do Dataset

Antes de treinar o modelo, é necessário criar um conjunto de dados. Execute o script `createdataset.py` para criar o conjunto de dados. Certifique-se de ajustar os caminhos de arquivos e pastas conforme necessário. O arquivo YAML já contém a estrutura da pasta para criar o conjunto de dados.

```bash
python createdataset.py
```

## Treinamento do Modelo

Por fim, para treinar o modelo, execute o arquivo `main.py`. Isso treinará o modelo de detecção de keypoints usando YOLOv8.

```bash
python main.py
```

Certifique-se de ajustar quaisquer outras configurações específicas do seu modelo, como hiperparâmetros, no arquivo `main.py`.

Isso deve configurar seu ambiente e permitir que você treine um modelo de detecção de keypoints usando YOLOv8, MediaPipe e OpenCV em Python.
```

Certifique-se de que você ajustou os caminhos de arquivo e as configurações do modelo conforme necessário antes de executar os comandos.
