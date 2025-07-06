import cv2
import numpy as np

# Carregar imagem colorida
imagem = cv2.imread('/home/pablo/Área de Trabalho/Projetos.py/PDI_02/polens.jpg')
if imagem is None:
    print("Erro: imagem não encontrada!")
    exit()

# Converter para tons de cinza
imagem_suavizada = cv2.GaussianBlur(imagem,(7, 7), 1.5)
cv2.imshow("GausBlur", imagem_suavizada)
imagem_suavizada = cv2.medianBlur(imagem_suavizada,5)
cv2.imshow("medianBlur", imagem_suavizada)
imagem_cinza = cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2GRAY)
cv2.imshow("escalaCinza", imagem_cinza)

# THRESH_BINARY_INV: Inverte para que os objetos fiquem brancos (255) e o fundo preto (0).
_, imagem_binaria_otsu = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# Elemento estruturante
kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

# Fechamento e dilatações para preencher buracos
preencher_furos = cv2.morphologyEx(imagem_binaria_otsu, cv2.MORPH_CLOSE, kernel1, iterations=4)
preencher_furos = cv2.dilate(preencher_furos, kernel1, iterations=4)

# Erosões e aberturas para separar objetos colados
separar_grao = cv2.erode(preencher_furos, kernel1, iterations=7)
separar_grao = cv2.morphologyEx(separar_grao, cv2.MORPH_OPEN, kernel1, iterations=10)


# Encontrar contornos
contornos, _ = cv2.findContours(separar_grao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calcula área média de grão individual a partir de contornos pequenos
areas = [cv2.contourArea(c) for c in contornos if 300 < cv2.contourArea(c) < 3000]  # ajuste de range
area_media = np.median(areas) if areas else 2000  # valor de fallback

# Estimar quantidade total
total_estimado = 0
imagem_resultado = cv2.cvtColor(separar_grao, cv2.COLOR_GRAY2BGR)

for cnt in contornos:
    area = cv2.contourArea(cnt)
    if area < 200:  # ruído
        continue

    estimado = int(round(area / area_media))
    total_estimado += estimado

    # Definir cor do retângulo pela quantidade estimada
    if estimado == 1:
        cor = (0, 255, 0)
    elif estimado == 2:
        cor = (255, 255, 0)
    else:
        cor = (0, 0, 255)

    # Desenhar retângulo nas imagens
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(imagem_resultado, (x, y), (x+w, y+h), cor, 2)
    cv2.rectangle(imagem, (x, y), (x+w, y+h), cor, 2)

    # Posição central do retângulo
    x_centro = x + w // 2
    y_centro = y + h // 2

    # Tamanho do texto para centralizar corretamente
    texto = f"{estimado}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.6
    espessura = 2
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, font, escala, espessura)

    # Posição ajustada para centralizar texto
    pos_x = x_centro - largura_texto // 2
    pos_y = y_centro + altura_texto // 2

    # Escrever quantidade estimada no centro do retângulo
    cv2.putText(imagem_resultado, texto, (pos_x, pos_y), font, escala, cor, espessura)
    cv2.putText(imagem, texto, (pos_x, pos_y), font, escala, cor, espessura)


cv2.imshow("binarizada", imagem_binaria_otsu)
cv2.imshow("preencher buracos", preencher_furos)
cv2.imshow("separar graos ", separar_grao)
cv2.imshow(f"Contagem Estimada: {total_estimado}", imagem_resultado)
cv2.imshow(f"Contagem: {total_estimado}", imagem)

print(f"Área média estimada de 1 grão: {area_media:.2f} px")
print(f"\n Total estimado de grãos: {total_estimado}")




cv2.waitKey(0)
cv2.destroyAllWindows()
