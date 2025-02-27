# TreinadorVisaoDeMaquina

O **TreinadorVisaoDeMaquina** é uma ferramenta de visão computacional baseada em aprendizado de máquina, inspirada no Teachable Machine, que permite aos usuários treinar e utilizar modelos de reconhecimento de objetos personalizados. Este projeto foi desenvolvido para rodar em computadores de escritório com especificações modestas, usando uma interface gráfica amigável para capturar imagens, treinar modelos, e visualizar resultados em tempo real.

## Descrição

O projeto utiliza o framework PyTorch (via `ultralytics` para YOLOv8) e TensorFlow/Keras para criar um sistema de aprendizado de máquina interativo. Ele inclui:
- **Captura de Imagens**: Usando uma webcam para coletar dados de treinamento para classes personalizadas (ex.: "Controle", "Fundo").
- **Treinamento de Modelos**: Treina um modelo de classificação de imagens com base nos dados coletados.
- **Pré-visualização**: Mostra o feed da webcam com barras de porcentagem indicando a confiança das previsões para cada classe.
- **Interface Gráfica**: Construída com `tkinter` e `ttkbootstrap`, com um design limpo e funcional.

## Requisitos

- Python 3.11 ou superior
- As seguintes bibliotecas (instale com `pip`):
  ```bash
  pip install ttkbootstrap opencv-python ultralytics tensorflow pillow numpy scikit-learn albumentations