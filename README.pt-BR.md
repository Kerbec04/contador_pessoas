# 👥 Contador de Pessoas

[English](README.md) · **Português**

[![Python](https://img.shields.io/badge/Python_3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111F68?style=for-the-badge)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)]()

Contagem de pessoas em imagens, vídeos, webcam e streams ao vivo usando visão computacional (YOLO,
da Ultralytics). O script detecta a classe **`person`** do dataset COCO, conta as pessoas e salva o
resultado anotado. Projeto desenvolvido no IFSP.

## 📈 Resultados

Numa foto de multidão em um show, os padrões originais (`imgsz=640`, `conf=0.3`) detectavam apenas
**9 pessoas**. O maior erro vinha da resolução de inferência baixa e da confiança alta, não do
tamanho do modelo. Com os novos padrões, `imgsz=1280` e `conf=0.2`, a contagem foi para
**cerca de 30 pessoas** com o mesmo modelo nano.

## 🚀 Funcionalidades

- **Detecção automática da entrada** pela `--fonte`:

| Fonte | Modo | Exemplo |
| --- | --- | --- |
| Arquivo de imagem | imagem | `--fonte foto.jpg` |
| Pasta com imagens | pasta | `--fonte ./imagens/` |
| Arquivo de vídeo | vídeo | `--fonte video.mp4` |
| Índice de webcam | webcam | `--fonte 0` |
| URL de stream (DroidCam/IP) | stream | `--fonte http://IP:4747/video` |

- **Pessoas únicas em vídeo** com rastreamento ByteTrack (`--tracking`), em vez de contar por frame
- **Registro de quem chegou e saiu:** salva o rosto de cada pessoa rastreada de tempo em tempo,
  agrupado por ID, mais um `presenca.csv` com horário, número de pessoas e IDs ativos
- **Saídas:** imagens e vídeo anotados e um CSV com a contagem por frame

## ⚙️ Instalação

Requer **Python 3.12+**.

```bash
pip install -r requirements.txt
# ou, com uv
uv sync
```

Na primeira execução, o modelo (`yolo26n.pt`) é baixado automaticamente.

## 💻 Uso

```bash
# imagem única, com resultado salvo em pasta
python contador_pessoas.py --fonte foto.jpg --saida ./resultados/

# multidão difícil: mais resolução e menos confiança detectam mais gente
python contador_pessoas.py --fonte foto.jpg --imgsz 1536 --conf 0.15

# vídeo contando pessoas únicas com rastreamento
python contador_pessoas.py --fonte video.mp4 --tracking --saida ./resultados/

# webcam ao vivo (pressione 'q' para parar)
python contador_pessoas.py --fonte 0 --tracking

# ao vivo, salvando rostos a cada 10 s para ver quem chegou ou saiu
python contador_pessoas.py --fonte 0 --tracking --snapshots ./rostos/ --intervalo 10
```

## 🎛️ Opções

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--fonte` | obrigatório | Imagem, pasta, vídeo, índice de webcam (`0`, `1`, …) ou URL de stream |
| `--modelo` | `yolo26n.pt` | Modelo YOLO. `n` = nano (rápido); `s`/`m` são mais precisos e mais lentos |
| `--conf` | `0.2` | Confiança mínima (0 a 1). Menor detecta mais, com mais falsos positivos |
| `--imgsz` | `1280` | Resolução de inferência. Maior detecta pessoas menores ou distantes |
| `--iou` | `0.6` | Limiar de IoU do NMS. Ajuda a separar pessoas muito próximas |
| `--max-det` | `1000` | Máximo de detecções por imagem |
| `--tracking` | desligado | Em vídeo, conta **pessoas únicas** (ByteTrack) em vez de por frame |
| `--snapshots` | nenhum | Pasta para salvar recortes de rosto periodicamente |
| `--intervalo` | `5` | Segundos entre cada salvamento de rostos |
| `--saida` | nenhum | Pasta para o resultado anotado e o CSV de contagem por frame |

## 📂 Saídas

- **Imagem ou pasta:** `anotada_<nome>.jpg` com as caixas e a contagem
- **Vídeo ou webcam:** `video_anotado.mp4` e `contagem_por_frame.csv` (`frame`, `pessoas_no_frame`, `unicas_acumuladas`)
- **Snapshots:** recortes de rosto (por ID, com `--tracking`) e `presenca.csv`

## ⚠️ Limitações Conhecidas

- Em oclusão severa (pessoas totalmente cobertas por outras), a contagem por detecção sempre
  subestima um pouco. É esperado em qualquer detector baseado em caixas.
- O ID vem do rastreador, não de reconhecimento facial: se a pessoa sai do quadro e volta, ou fica
  muito tempo tapada, ela pode ganhar um ID novo.
- O recorte do rosto é uma aproximação (topo da caixa do corpo) e pode sair torto quando a pessoa
  olha para baixo.
- Recortes de rosto são dados pessoais. Use a função de snapshots só onde as pessoas foram avisadas.
