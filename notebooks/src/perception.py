# =============================================================
# src/perception.py
# Motor de percepción del agente — YOLO + SegFormer + ResNet
# Equipo8-Grupo8-SIC-2025
# =============================================================

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from pathlib import Path
from typing import List
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


# ----------------------------------------------------------
# Arquitectura ResNet18+SPP — copiada del notebook 02
# ----------------------------------------------------------

class _SPPBlock(nn.Module):
    def __init__(self, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        B, C, H, W = x.shape
        pooled = [x]
        for size in self.pool_sizes:
            p = F.adaptive_avg_pool2d(x, output_size=size)
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            pooled.append(p)
        return torch.cat(pooled, dim=1)   # (B, C*4, H, W)


class _ResNet18SPP(nn.Module):
    def __init__(self, pool_sizes=(1, 2, 4)):
        super().__init__()
        backbone    = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.spp    = _SPPBlock(pool_sizes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.spp(x)   # (B, 2048, H/32, W/32)


class PerceptionEngine:
    """
    Ejecuta los modelos de percepción según el nivel de riesgo
    actual del AgentState (strategy pattern).

    Solo los modelos en active_models se cargan y ejecutan;
    el offloading lo gestiona el agente exterior.
    """

    def __init__(self, registry):
        """
        Parámetros
        ----------
        registry : ModelRegistry
            Registro ya configurado con los 6 modelos del pipeline.
        """
        self.registry = registry

        # Procesador SegFormer — solo normalización/resize, independiente
        # del checkpoint concreto cargado en el registry.
        self.seg_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            use_safetensors=True,
        )

        # Constantes de normalización ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

        # Tamaños de entrada por modelo
        self.SIZE_YOLO = 416
        self.SIZE_SEG  = 320
        self.SIZE_RES  = (384, 288)   # (ancho, alto)

    # ----------------------------------------------------------
    # Preprocesamiento
    # ----------------------------------------------------------

    def _apply_clahe(self, frame_bgr: np.ndarray, clahe_mode: str) -> np.ndarray:
        """
        Aplica CLAHE condicional según el modo de condición lumínica.

        Modos:
          "night"   → clipLimit=3.0, tileGridSize=(8,8)
          "weather" → clipLimit=2.5, tileGridSize=(8,8)
          "normal"  → sin modificar
        """
        if clahe_mode == "normal":
            return frame_bgr

        clip = 3.0 if clahe_mode == "night" else 2.5
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _preprocess_seg(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocesa frame para SegFormer manual (fallback).
        Resize → RGB → normalización ImageNet → CHW → batch dim.
        """
        img = cv2.resize(frame_bgr, (self.SIZE_SEG, self.SIZE_SEG))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor

    def _preprocess_res(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocesa frame para ResNet18SPP.
        Resize 384×288 → RGB → normalización ImageNet → CHW → batch dim.
        """
        w, h = self.SIZE_RES
        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor

    # ----------------------------------------------------------
    # Inferencia principal
    # ----------------------------------------------------------

    def run(
        self,
        frame_bgr: np.ndarray,
        active_models: List[str],
        clahe_mode: str = "normal",
    ) -> dict:
        """
        Ejecuta los modelos indicados en active_models sobre el frame.

        Parámetros
        ----------
        frame_bgr : np.ndarray
            Frame en formato BGR (OpenCV).
        active_models : List[str]
            Lista de nombres de modelos a ejecutar.
        clahe_mode : str
            "normal" | "night" | "weather"

        Retorna
        -------
        dict con claves:
            frame_clahe, boxes, mask, features, risk_score, clahe_mode
        """
        result = {
            "frame_clahe": None,
            "boxes":       None,
            "mask":        None,
            "features":    None,
            "risk_score":  None,
            "clahe_mode":  clahe_mode,
        }

        # 1. CLAHE condicional
        frame_clahe = self._apply_clahe(frame_bgr, clahe_mode)
        result["frame_clahe"] = frame_clahe

        # 2. YOLO
        if "yolo" in active_models:
            model = self.registry.load_model("yolo")
            boxes = model(frame_clahe, imgsz=self.SIZE_YOLO, verbose=False)[0]
            result["boxes"] = boxes

        # 3. SegFormer
        if "segformer" in active_models:
            model = self.registry.load_model("segformer")

            # Si loader_torch devolvió un state_dict, reconstruir arquitectura
            if isinstance(model, dict):
                seg_model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
                    num_labels=19,
                    ignore_mismatched_sizes=True,
                    use_safetensors=True,
                )
                seg_model.load_state_dict(model)
                seg_model.eval()
                seg_model = seg_model.to(self.registry.device)
                # Reemplazar en registry para no reconstruir en cada frame
                self.registry.registry["segformer"]["model"] = seg_model
                self.registry.registry["segformer"]["on_gpu"] = True
                self.registry.loaded_on_gpu.add("segformer")
                model = seg_model

            frame_rgb = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2RGB)
            inputs = self.seg_processor(images=frame_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.registry.device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            H, W = frame_bgr.shape[:2]
            logits_up = F.interpolate(
                outputs.logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            mask = logits_up.argmax(dim=1).squeeze(0).cpu()   # HxW int64
            result["mask"] = mask

        # 4. ResNet
        if "resnet" in active_models:
            model = self.registry.load_model("resnet")

            # Si loader_torch devolvió un state_dict, reconstruir arquitectura
            if isinstance(model, dict):
                res_model = _ResNet18SPP(pool_sizes=(1, 2, 4))
                res_model.load_state_dict(model)
                res_model.eval()
                res_model = res_model.to(self.registry.device)
                self.registry.registry["resnet"]["model"]  = res_model
                self.registry.registry["resnet"]["on_gpu"] = True
                self.registry.loaded_on_gpu.add("resnet")
                model = res_model

            tensor = self._preprocess_res(frame_clahe).to(self.registry.device)
            with torch.no_grad():
                features = model(tensor)
            result["features"] = features.cpu()

        # 5. GCN + LSTM + BNN — placeholder hasta siguiente iteración
        if any(m in active_models for m in ("gcn", "lstm", "bnn")):
            result["risk_score"] = 0.5

        return result


# =============================================================
# Bloque de prueba — ejecutar directamente para verificar
# =============================================================

if __name__ == "__main__":
    import sys
    import torch.nn as nn
    import torchvision.models as models
    from ultralytics import YOLO
    from transformers import SegformerForSemanticSegmentation
    from model_manager import ModelRegistry

    # ----------------------------------------------------------
    # Arquitecturas necesarias para reconstruir los modelos
    # desde state_dict
    # ----------------------------------------------------------

    class SPPBlock(nn.Module):
        def __init__(self, pool_sizes=[1, 2, 4]):
            super().__init__()
            self.pool_sizes = pool_sizes

        def forward(self, x):
            B, C, H, W = x.shape
            pooled = [x]
            for size in self.pool_sizes:
                p = F.adaptive_avg_pool2d(x, output_size=size)
                p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
                pooled.append(p)
            return torch.cat(pooled, dim=1)

    class ResNet18SPP(nn.Module):
        def __init__(self, pool_sizes=[1, 2, 4]):
            super().__init__()
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                        backbone.relu, backbone.maxpool)
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.spp    = SPPBlock(pool_sizes=pool_sizes)
            self.out_channels = 512 * (1 + len(pool_sizes))

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.spp(x)
            return x

    # ----------------------------------------------------------
    # Loaders que devuelven objetos nn.Module listos para .to()
    # ----------------------------------------------------------

    def loader_yolo(path):
        return YOLO(str(path))

    def loader_segformer(path):
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
            num_labels=19,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model

    def loader_resnet(path):
        model = ResNet18SPP(pool_sizes=[1, 2, 4])
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model

    def loader_torch(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    # ----------------------------------------------------------
    # Rutas
    # ----------------------------------------------------------

    BASE = Path("/mnt/bigdata/pipeline_samsung/Equipo8-Grupo8-SIC-2025-/notebooks")
    CKPT = BASE / "data/spatial_outputs"

    # ----------------------------------------------------------
    # Imagen de prueba real
    # ----------------------------------------------------------

    test_imgs = list((BASE / "data/curated").rglob("*_curated.jpg"))
    if not test_imgs:
        print("[ERROR] No se encontró ninguna imagen _curated.jpg en data/curated/")
        sys.exit(1)

    img_path = test_imgs[0]
    frame_bgr = cv2.imread(str(img_path))
    print(f"[Test] Imagen cargada : {img_path.name}  shape={frame_bgr.shape}")

    # ----------------------------------------------------------
    # Registry con loaders reales
    # ----------------------------------------------------------

    registry = ModelRegistry(vram_limit_gb=5.0)
    registry.register_model("yolo",      BASE / "yolov8n.pt",                                     loader_yolo,      0.6)
    registry.register_model("segformer", CKPT / "segformer_best.pt",                              loader_segformer, 1.2)
    registry.register_model("resnet",    CKPT / "resnet_best.pt",                                 loader_resnet,    0.3)
    registry.register_model("gcn",       CKPT / "checkpoints_03/gcn_best.pt",                     loader_torch,     0.1)
    registry.register_model("lstm",      CKPT / "checkpoints_03/lstm_best.pt",                    loader_torch,     0.1)
    registry.register_model("bnn",       CKPT / "prediction_outputs/checkpoints_bnn/bnn_best.pt", loader_torch,     0.2)

    engine = PerceptionEngine(registry)

    # ----------------------------------------------------------
    # Prueba 1 — nivel "low" → solo YOLO
    # ----------------------------------------------------------

    print("\n" + "="*55)
    print("PRUEBA 1 — nivel 'low'  →  active_models=['yolo']")
    print("="*55)
    out = engine.run(frame_bgr, active_models=["yolo"], clahe_mode="normal")

    print(f"clahe_mode   : {out['clahe_mode']}")
    boxes = out["boxes"]
    n_det = len(boxes.boxes) if boxes is not None and hasattr(boxes, "boxes") else 0
    print(f"boxes        : {n_det} detecciones YOLO")
    print(f"mask         : {out['mask']}")
    print(f"features     : {out['features']}")
    print(f"risk_score   : {out['risk_score']}")

    # ----------------------------------------------------------
    # Prueba 2 — nivel "high" → YOLO + SegFormer + ResNet
    # ----------------------------------------------------------

    print("\n" + "="*55)
    print("PRUEBA 2 — nivel 'high'  →  active_models=['yolo','segformer','resnet']")
    print("="*55)
    out2 = engine.run(
        frame_bgr,
        active_models=["yolo", "segformer", "resnet"],
        clahe_mode="night",
    )

    print(f"clahe_mode   : {out2['clahe_mode']}")
    boxes2 = out2["boxes"]
    n_det2 = len(boxes2.boxes) if boxes2 is not None and hasattr(boxes2, "boxes") else 0
    print(f"boxes        : {n_det2} detecciones YOLO")

    mask = out2["mask"]
    if mask is not None:
        print(f"mask         : shape={tuple(mask.shape)}  "
              f"clases únicas={mask.unique().tolist()}")
    else:
        print("mask         : None")

    feats = out2["features"]
    if feats is not None:
        print(f"features     : shape={tuple(feats.shape)}")
    else:
        print("features     : None")

    print(f"risk_score   : {out2['risk_score']}")

    print("\n[✓] PerceptionEngine.run() verificado en nivel 'low' y 'high'.")
