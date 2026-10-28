# 👥 Contador de Pessoas

**English** · [Português](README.pt-BR.md)

[![Python](https://img.shields.io/badge/Python_3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111F68?style=for-the-badge)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)]()

People counting in images, videos, webcams and live streams using computer vision (YOLO from
Ultralytics). The script detects the COCO **`person`** class, counts people and saves annotated
output. Built as a course project at IFSP.

## 📈 Results

On a crowd photo from a concert, the original settings (`imgsz=640`, `conf=0.3`) detected only
**9 people**. The biggest error came from low inference resolution and a high confidence threshold,
not from model size. Raising the defaults to `imgsz=1280` and `conf=0.2` brought the count to
**about 30 people** with the same nano model.

## 🚀 Features

- **Automatic input detection** from `--fonte` (source):

| Source | Mode | Example |
| --- | --- | --- |
| Image file | image | `--fonte photo.jpg` |
| Image folder | folder | `--fonte ./images/` |
| Video file | video | `--fonte video.mp4` |
| Webcam index | webcam | `--fonte 0` |
| Stream URL (DroidCam/IP) | stream | `--fonte http://IP:4747/video` |

- **Unique people in video** with ByteTrack tracking (`--tracking`), instead of a per-frame count
- **Arrival and departure log:** saves a face crop of each tracked person at a set interval, grouped
  by ID, plus a `presenca.csv` with time, people count and active IDs
- **Exports:** annotated images and video, and a per-frame count CSV

## ⚙️ Installation

Requires **Python 3.12+**.

```bash
pip install -r requirements.txt
# or, with uv
uv sync
```

The model (`yolo26n.pt`) is downloaded automatically on the first run.

## 💻 Usage

```bash
# single image, result saved to a folder
python contador_pessoas.py --fonte photo.jpg --saida ./results/

# hard crowd: higher resolution and lower confidence detect more people
python contador_pessoas.py --fonte photo.jpg --imgsz 1536 --conf 0.15

# video, counting unique people with tracking
python contador_pessoas.py --fonte video.mp4 --tracking --saida ./results/

# live webcam (press 'q' to stop)
python contador_pessoas.py --fonte 0 --tracking

# live, saving face crops every 10 s to see who arrived or left
python contador_pessoas.py --fonte 0 --tracking --snapshots ./faces/ --intervalo 10
```

## 🎛️ Options

| Argument | Default | Description |
| --- | --- | --- |
| `--fonte` | required | Image, folder, video, webcam index (`0`, `1`, …) or stream URL |
| `--modelo` | `yolo26n.pt` | YOLO model. `n` = nano (fast); `s`/`m` are more accurate and slower |
| `--conf` | `0.2` | Minimum confidence (0 to 1). Lower finds more people, with more false positives |
| `--imgsz` | `1280` | Inference resolution. Higher finds smaller or distant people |
| `--iou` | `0.6` | NMS IoU threshold. Helps separate people standing close together |
| `--max-det` | `1000` | Maximum detections per image |
| `--tracking` | off | In video, counts **unique people** (ByteTrack) instead of per frame |
| `--snapshots` | none | Folder for periodic face crops |
| `--intervalo` | `5` | Seconds between face crop saves |
| `--saida` | none | Folder for annotated output and the per-frame count CSV |

## 📂 Outputs

- **Image or folder:** `anotada_<name>.jpg` with boxes and the count
- **Video or webcam:** `video_anotado.mp4` and `contagem_por_frame.csv` (`frame`, `pessoas_no_frame`, `unicas_acumuladas`)
- **Snapshots:** face crops (by ID when `--tracking` is on) and `presenca.csv`

## ⚠️ Known Limitations

- With heavy occlusion (people fully hidden behind others), detection-based counting always
  undercounts somewhat. This is expected for any box-based detector.
- IDs come from the tracker, not from face recognition: a person who leaves the frame and comes
  back, or stays hidden for long, may get a new ID.
- The face crop is an approximation (top of the body box) and can be off when the person looks down.
- Face crops are personal data. Only use the snapshot feature where people have been informed.
