"""
Contador de pessoas em multidao com visao computacional.

Funciona em 4 modos de entrada, detectados automaticamente pela --fonte:
  - imagem unica  ->  caminho/foto.jpg
  - pasta com varias imagens  ->  caminho/da/pasta/
  - video (arquivo)  ->  caminho/video.mp4
  - webcam ao vivo  ->  --fonte 0

Usa YOLO (Ultralytics) pre-treinado no dataset COCO, onde a classe 'person'
tem id = 0.

Contagem em multidao (importante):
    O maior erro de contagem vem da RESOLUCAO de inferencia e da confianca,
    nao do tamanho do modelo. Pessoas pequenas/distantes so aparecem quando a
    imagem e processada em alta resolucao. Por isso os padroes aqui sao
    imgsz=1280 e conf=0.2 (em vez dos 640/0.25 do YOLO). Numa foto de show,
    isso levou a deteccao de 9 -> ~30 pessoas com o mesmo modelo nano.
    Ainda assim, oclusao severa (pessoas cobertas por outras) sempre subestima
    um pouco; isso e esperado e deve ser discutido no relatorio.

Exemplos de uso:
    python contador_pessoas.py --fonte teste.jpg
    python contador_pessoas.py --fonte teste.jpg --imgsz 1536 --conf 0.15
    python contador_pessoas.py --fonte ./imagens/ --saida ./resultados/
    python contador_pessoas.py --fonte video.mp4 --tracking
    python contador_pessoas.py --fonte 0 --tracking          # webcam

Instalacao:
    pip install ultralytics opencv-python
"""

import argparse
import csv
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Extensoes reconhecidas
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

# No dataset COCO a classe 'person' tem id 0. Filtrar so essa classe deixa
# a deteccao mais rapida e o desenho mais limpo.
PERSON_CLASS_ID = 0

VERDE = (0, 255, 0)


def desenhar_contagem(imagem, texto):
    """Escreve a contagem num cantinho da imagem, com fundo para legibilidade."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala, espessura = 0.8, 2
    (largura_txt, altura_txt), _ = cv2.getTextSize(texto, fonte, escala, espessura)
    cv2.rectangle(imagem, (5, 5), (15 + largura_txt, 20 + altura_txt), (0, 0, 0), -1)
    cv2.putText(imagem, texto, (10, 18 + altura_txt), fonte, escala, VERDE, espessura)
    return imagem


def recortar_rosto(frame, caixa):
    """
    Recorta a regiao do rosto a partir da caixa (x1,y1,x2,y2) de uma pessoa.

    O YOLO/COCO detecta o corpo inteiro, nao o rosto. Como as pessoas estao
    quase sempre em pe e de frente, aproximamos o rosto pela faixa de cima da
    caixa (topo ~45% da altura). E uma heuristica simples, boa o bastante para
    identificar visualmente quem e quem no registro.
    """
    x1, y1, x2, y2 = (int(v) for v in caixa)
    altura_caixa = y2 - y1
    y_rosto = y1 + int(0.45 * altura_caixa)  # pega cabeca + ombros

    # Garante que os limites ficam dentro da imagem
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y_rosto = max(0, y1), min(h, y_rosto)
    if x2 <= x1 or y_rosto <= y1:
        return None
    return frame[y1:y_rosto, x1:x2]


def salvar_rostos(frame, resultado, pasta, usar_tracking, momento):
    """
    Salva o recorte do rosto de cada pessoa detectada no frame.

    Com tracking: agrupa por ID (pasta rostos/id_<id>/), montando uma galeria
    de cada pessoa ao longo do tempo -> facil ver quem chegou (ID novo) ou saiu
    (ID que parou de aparecer).

    Sem tracking: nao ha ID, entao salva tudo numa subpasta por horario
    (rostos/<horario>/pessoa_<n>.jpg).

    Retorna a lista de IDs ativos neste instante (vazia se sem tracking).
    """
    caixas = resultado.boxes.xyxy.cpu().numpy()
    ids = None
    if usar_tracking and resultado.boxes.id is not None:
        ids = resultado.boxes.id.int().cpu().tolist()

    carimbo = momento.strftime("%H-%M-%S")
    ids_ativos = []

    for i, caixa in enumerate(caixas):
        rosto = recortar_rosto(frame, caixa)
        if rosto is None or rosto.size == 0:
            continue

        if ids is not None:
            pessoa_id = ids[i]
            ids_ativos.append(pessoa_id)
            destino_dir = pasta / f"id_{pessoa_id}"
            destino_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino_dir / f"{carimbo}.jpg"), rosto)
        else:
            destino_dir = pasta / carimbo
            destino_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino_dir / f"pessoa_{i + 1}.jpg"), rosto)

    return ids_ativos


def contar_em_imagem(model, caminho, infer, pasta_saida):
    """Detecta pessoas numa imagem, salva a versao anotada e retorna a contagem."""
    resultado = model.predict(str(caminho), **infer)[0]

    contagem = len(resultado.boxes)
    anotada = resultado.plot()  # imagem BGR com as caixas desenhadas
    anotada = desenhar_contagem(anotada, f"Pessoas: {contagem}")

    if pasta_saida:
        pasta_saida.mkdir(parents=True, exist_ok=True)
        destino = pasta_saida / f"anotada_{caminho.name}"
        cv2.imwrite(str(destino), anotada)
        print(f"  -> salvo em {destino}")

    print(f"{caminho.name}: {contagem} pessoa(s)")
    return contagem


def contar_em_pasta(model, pasta, infer, pasta_saida):
    """Roda a contagem em todas as imagens de uma pasta."""
    imagens = sorted(p for p in pasta.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imagens:
        print(f"Nenhuma imagem encontrada em {pasta}")
        return

    total = 0
    for img in imagens:
        total += contar_em_imagem(model, img, infer, pasta_saida)
    print(f"\nTotal somado em {len(imagens)} imagens: {total} pessoa(s)")


class LeitorMJPEG:
    """
    Leitor de stream MJPEG via HTTP, usando so a biblioteca padrao.

    Serve de alternativa quando o cv2.VideoCapture nao consegue abrir a URL
    (erro tipico do DroidCam: "backend ... can't be used to capture by name").
    Le os bytes crus do stream, separa cada quadro JPEG pelos marcadores
    de inicio (FF D8) e fim (FF D9) e decodifica com o OpenCV. Funciona sempre
    que o video aparece no navegador, pois le exatamente os mesmos bytes.

    Imita a interface do cv2.VideoCapture (read / isOpened / release) para
    poder ser usado no mesmo laco sem mudar o resto do codigo.
    """

    def __init__(self, url, timeout=10):
        # Alguns servidores recusam requisicoes sem User-Agent de navegador.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        self.stream = urllib.request.urlopen(req, timeout=timeout)
        self.buffer = b""
        self._aberto = True

    def isOpened(self):
        return self._aberto

    def read(self):
        # Acumula bytes ate ter um JPEG completo (entre FF D8 e FF D9).
        while True:
            # 1) Tenta extrair um quadro do que ja esta no buffer
            inicio = self.buffer.find(b"\xff\xd8")
            fim = self.buffer.find(b"\xff\xd9", inicio + 2) if inicio != -1 else -1
            if inicio != -1 and fim != -1:
                jpg = self.buffer[inicio:fim + 2]
                self.buffer = self.buffer[fim + 2:]
                quadro = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if quadro is not None:
                    return True, quadro
                continue  # JPEG corrompido: descarta e tenta o proximo
            # 2) Sem quadro completo ainda: le mais bytes da rede
            try:
                chunk = self.stream.read(8192)
            except Exception:
                return False, None
            if not chunk:
                return False, None
            self.buffer += chunk

    def release(self):
        self._aberto = False
        try:
            self.stream.close()
        except Exception:
            pass


def abrir_captura(fonte, eh_url):
    """
    Tenta abrir a fonte com o cv2.VideoCapture; se for URL e falhar, cai no
    leitor MJPEG manual. Para URLs sem endpoint explicito, tenta tambem o
    /mjpegfeed do DroidCam, que costuma funcionar melhor que o /video.
    """
    if not eh_url:
        return cv2.VideoCapture(fonte)

    # 1) Tenta direto com o backend FFmpeg
    captura = cv2.VideoCapture(fonte, cv2.CAP_FFMPEG)
    if captura.isOpened():
        return captura

    # 2) Fallback: leitor MJPEG manual (a prova de falhas se abre no navegador)
    candidatos = [fonte]
    # Se a URL termina em /video, tenta tambem o endpoint cru e o "force"
    # (force sobrepoe uma conexao existente; o DroidCam serve 1 cliente por vez)
    if fonte.rstrip("/").endswith("/video"):
        base = fonte.rstrip("/")[: -len("/video")]
        candidatos.append(base + "/mjpegfeed")
        candidatos.append(base + "/video/force/640x480")
    for url in candidatos:
        try:
            leitor = LeitorMJPEG(url)
            ok, _ = leitor.read()
            if ok:
                print(f"  -> usando leitor MJPEG manual em {url}")
                leitor.release()
                return LeitorMJPEG(url)  # reabre para comecar do inicio
            leitor.release()
            print(f"  -> {url}: conectou mas nao veio quadro valido")
        except Exception as erro:
            # Mostra o motivo real em vez de engolir o erro silenciosamente
            print(f"  -> {url}: falhou ({type(erro).__name__}: {erro})")
    return captura  # devolve a captura fechada para disparar o erro adiante


def contar_em_video(model, fonte, infer, usar_tracking, pasta_saida,
                    pasta_rostos=None, intervalo_rostos=5):
    """
    Processa um video (ou webcam) frame a frame.

    Sem --tracking: conta as deteccoes de cada frame (a mesma pessoa e recontada
    em frames diferentes -> use para densidade instantanea por frame).

    Com --tracking: usa o ByteTrack para dar um ID a cada pessoa e conta IDs
    unicos -> serve para contar quantas pessoas distintas apareceram no total.

    Com --snapshots: a cada 'intervalo_rostos' segundos salva o rosto de cada
    pessoa (agrupado por ID, se houver tracking) e registra em presenca.csv quem
    estava presente naquele instante -> serve para ver quem chegou/saiu ao vivo.
    """
    eh_url = isinstance(fonte, str) and fonte.startswith(
        ("http://", "https://", "rtsp://")
    )
    captura = abrir_captura(fonte, eh_url)

    if not captura.isOpened():
        msg = f"Nao consegui abrir a fonte de video: {fonte}"
        if eh_url:
            msg += ("\nO video abre no navegador mas nem o OpenCV nem o leitor "
                    "manual conseguiram. Tente trocar o final da URL de /video "
                    "para /mjpegfeed (ex.: http://IP:4747/mjpegfeed?640x480).")
        raise RuntimeError(msg)

    # Com VideoCapture conseguimos consultar fps/tamanho; com o leitor manual
    # nao, entao usamos valores padrao e descobrimos o tamanho no 1o frame.
    if isinstance(captura, cv2.VideoCapture):
        fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
    else:
        fps = 25.0

    gravador = None        # criado de forma preguicosa, no 1o frame
    caminho_csv = None
    if pasta_saida:
        pasta_saida.mkdir(parents=True, exist_ok=True)
        caminho_csv = pasta_saida / "contagem_por_frame.csv"

    # Registro de presenca (snapshots de rosto de tempo em tempo)
    presenca = []  # linhas (horario, n_pessoas, ids_ativos)
    if pasta_rostos:
        pasta_rostos.mkdir(parents=True, exist_ok=True)
    proximo_snapshot = 0.0  # marca de tempo (segundos) do proximo salvamento

    ids_unicos = set()
    registro = []  # linhas (frame, contagem_no_frame, unicas_acumuladas)
    frame_idx = 0
    # Roda inferencia so a cada INFERENCIA_A_CADA frames para aliviar o CPU/GPU
    INFERENCIA_A_CADA = 5
    ultimo_resultado = None
    ultima_contagem = 0

    # Em stream de rede (WiFi) frames falham de vez em quando. Em vez de
    # encerrar no primeiro erro, toleramos uma sequencia de falhas e so
    # paramos se a fonte realmente cair (varias falhas seguidas).
    falhas_seguidas = 0
    max_falhas = 30  # ~1s de tentativas antes de desistir, num stream tipico

    print("Processando... (pressione 'q' na janela para parar)")
    while True:
        ok, frame = captura.read()
        if not ok:
            if eh_url and falhas_seguidas < max_falhas:
                falhas_seguidas += 1
                continue  # frame perdido momentaneo: tenta o proximo
            break  # arquivo terminou, ou stream caiu de vez
        falhas_seguidas = 0
        frame_idx += 1

        # Cria o gravador no 1o frame, ja sabendo o tamanho real da imagem
        if pasta_saida and gravador is None:
            altura, largura = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            destino_video = pasta_saida / "video_anotado.mp4"
            gravador = cv2.VideoWriter(
                str(destino_video), fourcc, fps, (largura, altura)
            )

        # Inferencia apenas nos frames multiplos de INFERENCIA_A_CADA
        if frame_idx % INFERENCIA_A_CADA == 1 or ultimo_resultado is None:
            if usar_tracking:
                resultado = model.track(
                    frame, persist=True, tracker="bytetrack.yaml", **infer
                )[0]
                if resultado.boxes.id is not None:
                    ids_unicos.update(resultado.boxes.id.int().tolist())
            else:
                resultado = model.predict(frame, **infer)[0]
            ultimo_resultado = resultado
            ultima_contagem = len(resultado.boxes)
            registro.append((frame_idx, ultima_contagem, len(ids_unicos)))
            anotado = resultado.plot()

            # Snapshot de rostos a cada 'intervalo_rostos' segundos. Fazemos so
            # nos frames de inferencia, pois so ai as caixas casam com o frame.
            agora = time.time()
            if pasta_rostos and agora >= proximo_snapshot:
                momento = datetime.now()
                ids_ativos = salvar_rostos(
                    frame, resultado, pasta_rostos, usar_tracking, momento
                )
                presenca.append((
                    momento.strftime("%Y-%m-%d %H:%M:%S"),
                    ultima_contagem,
                    " ".join(str(i) for i in sorted(ids_ativos)),
                ))
                rotulo = (f"IDs ativos: {sorted(ids_ativos)}" if usar_tracking
                          else f"{ultima_contagem} rosto(s)")
                print(f"  [snapshot {momento.strftime('%H:%M:%S')}] {rotulo}")
                proximo_snapshot = agora + intervalo_rostos
        else:
            # Frames intermediarios: exibe o frame atual sem rodar o modelo
            anotado = frame.copy()

        texto = f"No frame: {ultima_contagem}"
        if usar_tracking:
            texto += f" | Unicas: {len(ids_unicos)}"
        anotado = desenhar_contagem(anotado, texto)

        if gravador:
            gravador.write(anotado)
        cv2.imshow("Contagem de pessoas", anotado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    if gravador:
        gravador.release()
    cv2.destroyAllWindows()

    # Salva o historico de contagem por frame (util para o relatorio/graficos)
    if caminho_csv:
        with open(caminho_csv, "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["frame", "pessoas_no_frame", "unicas_acumuladas"])
            escritor.writerows(registro)
        print(f"  -> contagem por frame salva em {caminho_csv}")

    # Salva o registro de presenca dos snapshots de rosto
    if pasta_rostos and presenca:
        caminho_presenca = pasta_rostos / "presenca.csv"
        with open(caminho_presenca, "w", newline="") as f:
            escritor = csv.writer(f)
            escritor.writerow(["horario", "n_pessoas", "ids_ativos"])
            escritor.writerows(presenca)
        print(f"  -> rostos salvos em {pasta_rostos} "
              f"({len(presenca)} snapshots)")
        print(f"  -> registro de presenca salvo em {caminho_presenca}")

    if usar_tracking:
        print(f"\nTotal de pessoas unicas detectadas: {len(ids_unicos)}")
    else:
        media = sum(c for _, c, _ in registro) / max(len(registro), 1)
        print(f"\nMedia de pessoas por frame: {media:.1f}")


def identificar_modo(fonte):
    """Decide se a fonte e webcam, stream de rede, pasta, imagem ou video."""
    # Webcam por indice: "0", "1", "2"... (0 = embutida; DroidCam vira 1 ou 2)
    if fonte.isdigit():
        return "webcam"
    # Stream de rede (DroidCam por IP, cameras IP, RTSP)
    if fonte.startswith(("http://", "https://", "rtsp://")):
        return "stream"
    caminho = Path(fonte)
    if caminho.is_dir():
        return "pasta"
    sufixo = caminho.suffix.lower()
    if sufixo in IMAGE_EXTS:
        return "imagem"
    if sufixo in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Nao reconheci o tipo da fonte: {fonte}")


def main():
    parser = argparse.ArgumentParser(description="Contador de pessoas (imagem/video)")
    parser.add_argument("--fonte", required=True,
                        help="imagem, pasta, video, indice de webcam (0,1,2...) "
                             "ou URL de stream (ex.: DroidCam "
                             "http://IP:4747/video)")
    parser.add_argument("--modelo", default="yolo26n.pt",
                        help="modelo YOLO (ex.: yolo26n.pt, yolo11n.pt). "
                             "O 'n' = nano (rapido); use 's'/'m' para mais precisao")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="confianca minima da deteccao (0 a 1). Em multidao, "
                             "valores menores (0.15-0.2) detectam mais gente")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="resolucao de inferencia. Maior = detecta pessoas "
                             "menores/distantes (640 padrao do YOLO subestima "
                             "muito em multidao); tente 1280 ou 1536")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="limiar de IoU do NMS. Em multidao, valores um pouco "
                             "menores evitam mesclar pessoas coladas")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="maximo de deteccoes por imagem (padrao do YOLO e 300)")
    parser.add_argument("--tracking", action="store_true",
                        help="conta pessoas unicas em video (ByteTrack). "
                             "Recomendado junto com --snapshots para agrupar os "
                             "rostos por pessoa")
    parser.add_argument("--snapshots", default=None,
                        help="pasta onde salvar recortes de rosto de tempo em "
                             "tempo (para ver quem chegou/saiu no ao vivo). "
                             "Com --tracking, agrupa por ID da pessoa")
    parser.add_argument("--intervalo", type=float, default=5.0,
                        help="segundos entre cada salvamento de rostos "
                             "(padrao 5s)")
    parser.add_argument("--saida", default=None,
                        help="pasta para salvar resultados (opcional)")
    args = parser.parse_args()

    pasta_saida = Path(args.saida) if args.saida else None
    pasta_rostos = Path(args.snapshots) if args.snapshots else None

    # Parametros de inferencia comuns a todos os modos. Filtrar so a classe
    # 'person' deixa a deteccao mais rapida e o desenho mais limpo.
    infer = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    print(f"Carregando modelo {args.modelo}...")
    model = YOLO(args.modelo)  # baixa os pesos na primeira vez

    modo = identificar_modo(args.fonte)
    if modo == "imagem":
        contar_em_imagem(model, Path(args.fonte), infer, pasta_saida)
    elif modo == "pasta":
        contar_em_pasta(model, Path(args.fonte), infer, pasta_saida)
    elif modo == "video":
        contar_em_video(model, args.fonte, infer, args.tracking, pasta_saida,
                        pasta_rostos, args.intervalo)
    elif modo == "stream":
        # URL de rede (DroidCam por IP, camera IP) - passa a string direto
        contar_em_video(model, args.fonte, infer, args.tracking, pasta_saida,
                        pasta_rostos, args.intervalo)
    elif modo == "webcam":
        # Indice da camera: 0 = embutida, 1/2 = DroidCam (webcam virtual)
        contar_em_video(model, int(args.fonte), infer, args.tracking,
                        pasta_saida, pasta_rostos, args.intervalo)


if __name__ == "__main__":
    main()