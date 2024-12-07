import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Carregar o modelo MobileNetV2 pré-treinado
model = MobileNetV2(weights="imagenet")

def analisar_imagem(caminho_imagem):
    # Carregar e processar a imagem
    imagem = load_img(caminho_imagem, target_size=(224, 224))  # Redimensionar para 224x224
    imagem_array = img_to_array(imagem)
    imagem_array = np.expand_dims(imagem_array, axis=0)  # Adicionar dimensão do batch
    imagem_array = preprocess_input(imagem_array)

    # Fazer a previsão
    previsoes = model.predict(imagem_array)
    resultados = decode_predictions(previsoes, top=3)[0]  # Top 3 resultados

    print("\nPrevisões:")
    for classe, nome, probabilidade in resultados:
        print(f"{nome}: {probabilidade:.2f}")
    return resultados[0][1]  # Retorna o nome da classe com maior probabilidade

# Exemplo de uso
if __name__ == "__main__":
    print("Digite o caminho da imagem da flor:")
    caminho = input("> ")
    nome_flor = analisar_imagem(caminho)
    print(f"\nO tipo de flor mais provável é: {nome_flor}")
