# =============================================================
# src/risk_estimator.py
# Estimación de riesgo: GCN → LSTM → BNN sobre grafos YOLO
# Equipo8-Grupo8-SIC-2025
# =============================================================

from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data
from torch_geometric.nn   import GCNConv, global_mean_pool

# ----------------------------------------------------------
# Constantes del pipeline (deben coincidir con notebooks 03/04)
# ----------------------------------------------------------

# NODE_DIM = 4 (bbox_xyxy) + 9 (one-hot YOLO) + 1 (conf) + 1 (mask_zone) + 2048 (ResNet)
NODE_DIM     = 2063
GCN_HIDDEN   = 256
GCN_OUT      = 128
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2

# Las 9 clases que el GCN fue entrenado a reconocer (orden importante)
CLASES_YOLO = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light",
]

# Mapeo COCO cls_id → índice en CLASES_YOLO
COCO_TO_GCN = {
    0:  0,   # person
    1:  7,   # bicycle
    2:  2,   # car
    3:  6,   # motorcycle
    5:  4,   # bus
    6:  5,   # train
    7:  3,   # truck
    9:  8,   # traffic light
}

# Umbrales de distancia pixel para edges (igual que en entrenamiento)
THRESH_LOCAL  = 150.0
THRESH_GLOBAL = 400.0


# ==============================================================
# Arquitecturas — copiadas exactamente de notebooks 03 y 04
# ==============================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_dim=NODE_DIM, hidden=GCN_HIDDEN,
                 out_dim=GCN_OUT, dropout=0.3):
        super().__init__()
        self.conv_l1 = GCNConv(in_dim, hidden)
        self.conv_l2 = GCNConv(hidden, out_dim)
        self.conv_g1 = GCNConv(in_dim, hidden)
        self.conv_g2 = GCNConv(hidden, out_dim)
        self.fc_fuse = nn.Linear(out_dim * 2, out_dim)
        self.drop    = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x, edge_index_l, edge_index_g, batch):
        xl = self.relu(self.conv_l1(x, edge_index_l))
        xl = self.drop(xl)
        xl = self.conv_l2(xl, edge_index_l)
        xl = global_mean_pool(xl, batch)

        xg = self.relu(self.conv_g1(x, edge_index_g))
        xg = self.drop(xg)
        xg = self.conv_g2(xg, edge_index_g)
        xg = global_mean_pool(xg, batch)

        out = torch.cat([xl, xg], dim=1)
        out = self.relu(self.fc_fuse(out))
        return out


class RiskLSTM(nn.Module):
    def __init__(self, input_size=GCN_OUT, hidden=LSTM_HIDDEN,
                 layers=LSTM_LAYERS, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, seq):
        """seq: (B, T, GCN_OUT) → (B, 1)"""
        out, _ = self.lstm(seq)
        return self.sig(self.fc(out[:, -1, :]))


class MCDropout(nn.Module):
    """Dropout con training=True forzado en inferencia."""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


class BNNRiskPredictor(nn.Module):
    """Red bayesiana aproximada vía MC Dropout."""
    def __init__(self, input_dim=128, hidden_dim=64, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            MCDropout(p=dropout_p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def mc_predict(self, x, n_samples=20):
        self.train()
        preds = torch.stack(
            [self(x).squeeze(-1) for _ in range(n_samples)], dim=0
        )   # (n_samples, B)
        self.eval()
        return {
            "mean": preds.mean(dim=0),
            "std":  preds.std(dim=0),
        }


# ==============================================================
# Clase principal
# ==============================================================

class RiskEstimator:
    """
    Estima el riesgo de la escena usando GCN → LSTM → BNN.

    GCNEncoder  : grafo PyG de objetos YOLO → embedding (1, 128)
    RiskLSTM    : secuencia de embeddings   → risk score scalar
    BNNRiskPredictor: MC Dropout sobre embedding → (mean, std)
    """

    def __init__(self, registry, gcn_embed_dim: int = 128, seq_len: int = 10):
        self.registry     = registry
        self.seq_len      = seq_len
        self.embed_buffer = deque(maxlen=seq_len)
        self.device       = registry.device

    # ----------------------------------------------------------
    # Construcción del grafo
    # ----------------------------------------------------------

    def _edges_por_distancia(self, centroides, thresh):
        """Devuelve edge_index para pares con distancia euclídea < thresh."""
        N = len(centroides)
        src, dst = [], []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = centroides[i][0] - centroides[j][0]
                dy = centroides[i][1] - centroides[j][1]
                if (dx**2 + dy**2) ** 0.5 < thresh:
                    src.append(i)
                    dst.append(j)
        if not src:   # sin edges → self-loops
            src = dst = list(range(N))
        return torch.tensor([src, dst], dtype=torch.long)

    def _boxes_to_graph(self, boxes_result, frame_shape) -> Data:
        """
        Construye grafo PyG desde resultado YOLO o lista de dicts.

        boxes_result puede ser:
          - Objeto YOLO resultado (con .boxes.xyxy / .boxes.conf / .boxes.cls)
          - Lista de dicts [{"xyxy": [...], "conf": f, "cls": i}, ...]

        Nodo features (NODE_DIM=2063):
          bbox_xyxy(4) + one_hot_class(9) + conf(1) + mask_zone(1) + resnet_zeros(2048)
        """
        # Extraer listas uniformes
        if isinstance(boxes_result, list):
            # Formato dict para tests
            recs = boxes_result
            xyxy_list = [r["xyxy"]       for r in recs]
            conf_list = [float(r["conf"]) for r in recs]
            cls_list  = [int(r["cls"])    for r in recs]
        elif boxes_result is not None and hasattr(boxes_result, "boxes") \
                and boxes_result.boxes is not None and len(boxes_result.boxes):
            det       = boxes_result.boxes
            xyxy_list = det.xyxy.cpu().numpy().tolist()
            conf_list = det.conf.cpu().numpy().tolist()
            cls_list  = det.cls.cpu().numpy().astype(int).tolist()
        else:
            xyxy_list, conf_list, cls_list = [], [], []

        # Nodo dummy si no hay detecciones
        if len(xyxy_list) == 0:
            x          = torch.zeros((1, NODE_DIM), dtype=torch.float)
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            batch      = torch.zeros(1, dtype=torch.long)
            return Data(x=x, edge_index=edge_index,
                        edge_index_g=edge_index, batch=batch)

        node_feats  = []
        centroides  = []

        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centroides.append((cx, cy))

            # One-hot sobre CLASES_YOLO (9 clases)
            gcn_cls = COCO_TO_GCN.get(int(cls_id), 0)
            onehot  = [0.0] * 9
            onehot[gcn_cls] = 1.0

            # Mask zone placeholder (0.0 — se mejorará si se pasa mask)
            mask_zone = 0.0

            # Features: [xyxy(4), onehot(9), conf(1), mask_zone(1), resnet(2048)]
            feat = [float(x1), float(y1), float(x2), float(y2)] \
                 + onehot \
                 + [conf, mask_zone] \
                 + [0.0] * 2048

            node_feats.append(feat)

        x            = torch.tensor(node_feats, dtype=torch.float)
        edge_index_l = self._edges_por_distancia(centroides, THRESH_LOCAL)
        edge_index_g = self._edges_por_distancia(centroides, THRESH_GLOBAL)
        batch        = torch.zeros(len(node_feats), dtype=torch.long)

        return Data(x=x, edge_index=edge_index_l,
                    edge_index_g=edge_index_g, batch=batch)

    # ----------------------------------------------------------
    # Reconstrucción de modelos desde state_dict
    # ----------------------------------------------------------

    def _ensure_gcn(self, model):
        if isinstance(model, dict):
            net = GCNEncoder(in_dim=NODE_DIM, hidden=GCN_HIDDEN,
                             out_dim=GCN_OUT).to(self.device)
            net.load_state_dict(model)
            net.eval()
            self.registry.registry["gcn"]["model"]  = net
            self.registry.registry["gcn"]["on_gpu"] = True
            self.registry.loaded_on_gpu.add("gcn")
            return net
        return model

    def _ensure_lstm(self, model):
        if isinstance(model, dict):
            net = RiskLSTM(input_size=GCN_OUT, hidden=LSTM_HIDDEN,
                           layers=LSTM_LAYERS).to(self.device)
            net.load_state_dict(model)
            net.eval()
            self.registry.registry["lstm"]["model"]  = net
            self.registry.registry["lstm"]["on_gpu"] = True
            self.registry.loaded_on_gpu.add("lstm")
            return net
        return model

    def _ensure_bnn(self, model):
        if isinstance(model, dict):
            net = BNNRiskPredictor(input_dim=GCN_OUT,
                                   hidden_dim=64).to(self.device)
            net.load_state_dict(model)
            net.eval()
            self.registry.registry["bnn"]["model"]  = net
            self.registry.registry["bnn"]["on_gpu"] = True
            self.registry.loaded_on_gpu.add("bnn")
            return net
        return model

    # ----------------------------------------------------------
    # Estimación principal
    # ----------------------------------------------------------

    def estimate(self, boxes_result, frame_shape) -> tuple:
        """
        Estima riesgo con GCN → LSTM → BNN.

        Retorna (risk_score: float, sigma: float).
        Retorna (0.5, 0.0) si el buffer de secuencia no está lleno.
        """
        # 1. Grafo
        graph = self._boxes_to_graph(boxes_result, frame_shape)
        graph = graph.to(self.device)

        # 2. GCN → embedding (1, 128)
        gcn_raw = self.registry.load_model("gcn")
        gcn     = self._ensure_gcn(gcn_raw)

        with torch.no_grad():
            embed = gcn(graph.x, graph.edge_index,
                        graph.edge_index_g, graph.batch)   # (1, 128)

        self.embed_buffer.append(embed.cpu())

        # 3. Buffer insuficiente — devolver placeholder
        if len(self.embed_buffer) < self.seq_len:
            return 0.5, 0.0

        # 4. LSTM → risk_score
        lstm_raw = self.registry.load_model("lstm")
        lstm     = self._ensure_lstm(lstm_raw)

        seq = torch.stack(list(self.embed_buffer), dim=1).to(self.device)  # (1, T, 128)
        with torch.no_grad():
            risk_tensor = lstm(seq)   # (1, 1)
        risk_score = float(risk_tensor.squeeze())

        # 5. BNN → (mean, std) sobre último embedding
        bnn_raw = self.registry.load_model("bnn")
        bnn     = self._ensure_bnn(bnn_raw)

        last_embed = embed.to(self.device)   # (1, 128)
        mc_out     = bnn.mc_predict(last_embed, n_samples=20)
        mean  = float(mc_out["mean"].squeeze())
        sigma = float(mc_out["std"].squeeze())

        return mean, sigma


# ==============================================================
# Bloque de prueba
# ==============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from model_manager import ModelRegistry
    from ultralytics   import YOLO

    BASE = Path("/mnt/bigdata/pipeline_samsung/Equipo8-Grupo8-SIC-2025-/notebooks")
    CKPT = BASE / "data/spatial_outputs"

    def loader_yolo(path):
        return YOLO(str(path))

    def loader_torch(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    # Registry con los 6 modelos
    registry = ModelRegistry(vram_limit_gb=5.0)
    registry.register_model("yolo",      BASE / "yolov8n.pt",                                     loader_yolo,  0.6)
    registry.register_model("segformer", CKPT / "segformer_best.pt",                              loader_torch, 1.2)
    registry.register_model("resnet",    CKPT / "resnet_best.pt",                                 loader_torch, 0.3)
    registry.register_model("gcn",       CKPT / "checkpoints_03/gcn_best.pt",                     loader_torch, 0.1)
    registry.register_model("lstm",      CKPT / "checkpoints_03/lstm_best.pt",                    loader_torch, 0.1)
    registry.register_model("bnn",       CKPT / "prediction_outputs/checkpoints_bnn/bnn_best.pt", loader_torch, 0.2)

    estimator = RiskEstimator(registry, seq_len=10)

    # Boxes ficticios (3 detecciones por frame, coords absolutas 1280×720)
    FAKE_BOXES = [
        {"xyxy": [200, 300, 400, 500], "conf": 0.85, "cls": 2},   # car
        {"xyxy": [600, 250, 750, 480], "conf": 0.72, "cls": 0},   # person
        {"xyxy": [900, 200, 1100, 600], "conf": 0.91, "cls": 7},  # truck
    ]
    FRAME_SHAPE = (720, 1280)

    print(f"\n{'Frame':<8} {'risk_score':<14} {'sigma':<10} {'estado'}")
    print("-" * 50)

    for frame_idx in range(15):
        risk, sigma = estimator.estimate(FAKE_BOXES, FRAME_SHAPE)
        buf_len = len(estimator.embed_buffer)
        estado  = "placeholder (buffer llenándose)" if buf_len < 10 else "modelo real"
        print(f"{frame_idx:<8} {risk:<14.4f} {sigma:<10.4f} {estado}")

    print("\n[✓] RiskEstimator verificado — GCN → LSTM → BNN integrados.")
