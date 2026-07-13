# Contador de Pessoas

Contagem de pessoas em imagens, vídeos e webcam usando visão computacional
(YOLO / Ultralytics). Projeto do IFSP.

O script detecta a classe **`person`** do dataset COCO e mostra/salva a imagem
anotada com a contagem.

---

## Como funciona

Usa um modelo YOLO pré-treinado. O modo de entrada é detectado automaticamente
pela `--fonte`:

| Fonte                        | Modo      | Exemplo                                  |
|------------------------------|-----------|------------------------------------------|
| Arquivo de imagem            | imagem    | `--fonte foto.jpg`                       |
| Pasta com imagens            | pasta     | `--fonte ./imagens/`                     |
| Arquivo de vídeo             | vídeo     | `--fonte video.mp4`                      |
| Índice de webcam             | webcam    | `--fonte 0`                              |
| URL de stream (DroidCam/IP)  | stream    | `--fonte http://IP:4747/video`           |

---

## Instalação

Requer **Python 3.12+**.

```bash
pip install -r requirements.txt
```

Ou, se usar o [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Na primeira execução o modelo (`yolo26n.pt`) é baixado automaticamente.

---

## Uso

```bash
# imagem única
python contador_pessoas.py --fonte teste.jpg

# imagem com resultado salvo em pasta
python contador_pessoas.py --fonte teste.jpg --saida ./resultados/

# multidão difícil: mais resolução e menos confiança = detecta mais gente
python contador_pessoas.py --fonte teste.jpg --imgsz 1536 --conf 0.15

# pasta inteira de imagens
python contador_pessoas.py --fonte ./imagens/ --saida ./resultados/

# vídeo contando pessoas únicas (tracking)
python contador_pessoas.py --fonte video.mp4 --tracking --saida ./resultados/

# webcam ao vivo (pressione 'q' na janela para parar)
python contador_pessoas.py --fonte 0 --tracking

# ao vivo salvando rostos a cada 10s para ver quem chegou/saiu
python contador_pessoas.py --fonte 0 --tracking --snapshots ./rostos/ --intervalo 10
```

---

## Opções

| Argumento     | Padrão       | Descrição                                                                 |
|---------------|--------------|--------------------------------------------------------------------------|
| `--fonte`     | (obrigatório)| Imagem, pasta, vídeo, índice de webcam (`0`,`1`,…) ou URL de stream.      |
| `--modelo`    | `yolo26n.pt` | Modelo YOLO. `n` = nano (rápido); `s`/`m` dão mais precisão (mais lento). |
| `--conf`      | `0.2`        | Confiança mínima (0 a 1). Menor detecta mais, porém com mais falsos.      |
| `--imgsz`     | `1280`       | Resolução de inferência. Maior detecta pessoas menores/distantes.        |
| `--iou`       | `0.6`        | Limiar de IoU do NMS. Ajuda a separar pessoas coladas.                    |
| `--max-det`   | `1000`       | Máximo de detecções por imagem.                                          |
| `--tracking`  | desligado    | Em vídeo, conta **pessoas únicas** (ByteTrack) em vez de por frame.       |
| `--snapshots` | nenhum       | Pasta onde salvar recortes de rosto de tempo em tempo (ver seção abaixo). |
| `--intervalo` | `5`          | Segundos entre cada salvamento de rostos.                                |
| `--saida`     | nenhum       | Pasta para salvar imagem/vídeo anotado e o CSV de contagem por frame.     |

---

## Precisão em multidão (importante)

Numa foto de show, o script detectava **apenas 9 pessoas** com os padrões
originais (`imgsz=640`, `conf=0.3`). O maior erro **não** vinha do tamanho do
modelo, e sim de:

1. **Resolução de inferência baixa** — pessoas pequenas/distantes só aparecem em
   alta resolução. Os padrões agora são `imgsz=1280` e `conf=0.2`, o que levou a
   contagem para **~30 pessoas** com o mesmo modelo nano.
2. **Confiança alta demais** — em multidão vale baixar para `0.15`–`0.2`.

Dicas para melhorar ainda mais:

- Aumente a resolução: `--imgsz 1536` (ou `1920`, se a máquina aguentar).
- Baixe a confiança: `--conf 0.15`.
- Use um modelo maior: `--modelo yolo11m.pt` (mais lento, um pouco mais preciso).

> **Limite conhecido:** em oclusão severa (pessoas totalmente cobertas por
> outras) a contagem por detecção **sempre subestima** um pouco. Isso é esperado
> em qualquer detector baseado em caixas e deve ser mencionado no relatório.

---

## Registro de rostos ao vivo — quem chegou / quem saiu

Use `--snapshots` numa fonte ao vivo (webcam ou stream) para **salvar o rosto de
cada pessoa de tempo em tempo** e acompanhar quem entrou ou saiu:

```bash
python contador_pessoas.py --fonte 0 --tracking --snapshots ./rostos/ --intervalo 10
```

A cada `--intervalo` segundos, o script recorta a região do rosto de cada pessoa
detectada e salva. **Combine com `--tracking`**: aí os recortes são organizados
por ID, montando uma galeria de cada pessoa ao longo do tempo:

```text
rostos/
├── id_1/                  # cada pasta = uma pessoa (ID do tracking)
│   ├── 14-32-05.jpg       # o mesmo rosto salvo a cada intervalo
│   └── 14-32-15.jpg
├── id_2/
│   └── 14-32-05.jpg
└── presenca.csv           # horario, n_pessoas, ids_ativos
```

Como ler:

- **Chegou alguém** → aparece uma **pasta de ID nova** (e um ID novo em
  `presenca.csv`).
- **Saiu alguém** → um ID **para de ganhar** fotos novas / some da lista de
  `ids_ativos`.
- Sem `--tracking`, os recortes vão para subpastas por horário
  (`rostos/14-32-05/pessoa_1.jpg`), sem agrupar por pessoa.

> **Limite conhecido:** o ID vem do rastreador (ByteTrack), não de
> reconhecimento facial. Se a pessoa **sai do quadro e volta**, ou fica muito
> tempo tapada por outra, ela pode ganhar um **ID novo**. Para reconhecer a mesma
> pessoa de verdade nesses casos seria preciso reconhecimento facial (biblioteca
> extra). O recorte do rosto também é uma aproximação (topo da caixa do corpo),
> então pode sair torto quando a pessoa está olhando para baixo.

---

## Saídas geradas

Com `--saida`:

- **Imagem/pasta:** `anotada_<nome>.jpg` com as caixas e a contagem.
- **Vídeo/webcam:** `video_anotado.mp4` e `contagem_por_frame.csv`
  (colunas: `frame`, `pessoas_no_frame`, `unicas_acumuladas`).

Com `--snapshots`:

- Recortes de rosto (por ID, se houver `--tracking`) e `presenca.csv`.

---

## Estrutura do projeto

```text
Contador/
├── contador_pessoas.py   # script principal
├── teste.jpg             # imagem de exemplo (multidão)
├── requirements.txt      # dependências (pip)
├── pyproject.toml        # dependências (uv)
└── README.md
```
