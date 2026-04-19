# =============================================================
# src/alert_logger.py
# Observer del agente — logging de alertas con debouncing
# Equipo8-Grupo8-SIC-2025
# =============================================================

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class AlertLogger:
    """
    Recibe risk_score por frame, decide si emitir alerta con
    debouncing por nivel, y persiste logs en CSV y JSON.
    """

    CSV_HEADER = ["timestamp", "frame", "level", "risk_score", "uncertainty"]

    def __init__(
        self,
        csv_path: Path,
        json_path: Path,
        debounce_frames: int = 10,
    ):
        self.csv_path        = Path(csv_path)
        self.json_path       = Path(json_path)
        self.debounce_frames = debounce_frames

        self.last_alert_frame: dict = {"medium": -999, "high": -999}
        self.alert_counts: dict     = {"medium": 0, "high": 0}

        self.metrics: dict = {
            "total_frames":   0,
            "frames_low":     0,
            "frames_medium":  0,
            "frames_high":    0,
            "alerts_medium":  0,
            "alerts_high":    0,
            "latencias_ms":   [],
        }

        # Inicializar CSV con cabecera
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)

    # ----------------------------------------------------------
    # Actualización por frame
    # ----------------------------------------------------------

    def update(
        self,
        frame_count: int,
        level: str,
        risk_score: float,
        uncertainty: Optional[float] = None,
        timestamp_ms: Optional[float] = None,
    ) -> Optional[str]:
        """
        Procesa un frame y emite alerta si corresponde.

        Retorna el level como str si se emitió alerta, None si no.
        """
        # Métricas de frames por nivel
        if level == "low":
            self.metrics["frames_low"] += 1
        elif level == "medium":
            self.metrics["frames_medium"] += 1
        elif level == "high":
            self.metrics["frames_high"] += 1

        # Latencia opcional
        if timestamp_ms is not None:
            self.metrics["latencias_ms"].append(timestamp_ms)

        self.metrics["total_frames"] += 1

        # Debouncing y emisión de alerta
        alerta = None
        if level in ("medium", "high"):
            frames_desde_ultima = frame_count - self.last_alert_frame[level]
            if frames_desde_ultima >= self.debounce_frames:
                ts = datetime.now().isoformat(timespec="milliseconds")
                unc_str = f"{uncertainty:.4f}" if uncertainty is not None else ""

                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ts, frame_count, level,
                                     f"{risk_score:.4f}", unc_str])

                self.last_alert_frame[level]  = frame_count
                self.alert_counts[level]     += 1
                self.metrics[f"alerts_{level}"] += 1
                alerta = level

        return alerta

    # ----------------------------------------------------------
    # Persistencia de métricas
    # ----------------------------------------------------------

    def save_metrics(
        self,
        video_path: str = "",
        fps_real: float = 0.0,
    ) -> None:
        """Escribe video_metrics.json con el resumen de la sesión."""
        lats = self.metrics["latencias_ms"]
        lat_media = sum(lats) / len(lats) if lats else 0.0
        lat_max   = max(lats)             if lats else 0.0
        lat_min   = min(lats)             if lats else 0.0

        payload = {
            "video":   video_path,
            "fecha":   datetime.now().isoformat(),
            "total_frames": self.metrics["total_frames"],
            "fps_real": fps_real,
            "frames_por_nivel": {
                "low":    self.metrics["frames_low"],
                "medium": self.metrics["frames_medium"],
                "high":   self.metrics["frames_high"],
            },
            "alertas": {
                "medium": self.metrics["alerts_medium"],
                "high":   self.metrics["alerts_high"],
            },
            "latencia_ms": {
                "media": round(lat_media, 3),
                "max":   round(lat_max,   3),
                "min":   round(lat_min,   3),
            },
        }

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[AlertLogger] Métricas guardadas → {self.json_path}")

    # ----------------------------------------------------------
    # Resumen en terminal
    # ----------------------------------------------------------

    def summary(self) -> None:
        """Imprime resumen de la sesión."""
        total = self.metrics["total_frames"]
        pct   = lambda n: (n / total * 100) if total > 0 else 0.0

        lats      = self.metrics["latencias_ms"]
        lat_media = sum(lats) / len(lats) if lats else None

        print("\n[AlertLogger] ── Resumen de sesión ──────────────────────")
        print(f"  Total frames procesados : {total}")
        print(f"  Frames low              : {self.metrics['frames_low']:>5}  "
              f"({pct(self.metrics['frames_low']):.1f}%)")
        print(f"  Frames medium           : {self.metrics['frames_medium']:>5}  "
              f"({pct(self.metrics['frames_medium']):.1f}%)")
        print(f"  Frames high             : {self.metrics['frames_high']:>5}  "
              f"({pct(self.metrics['frames_high']):.1f}%)")
        print(f"  Alertas medium          : {self.metrics['alerts_medium']}")
        print(f"  Alertas high            : {self.metrics['alerts_high']}")
        if lat_media is not None:
            print(f"  Latencia media          : {lat_media:.2f} ms")
        print("[AlertLogger] ────────────────────────────────────────────")


# =============================================================
# Bloque de prueba — ejecutar directamente para verificar
# =============================================================

if __name__ == "__main__":
    BASE = Path("/mnt/bigdata/pipeline_samsung/Equipo8-Grupo8-SIC-2025-/notebooks")
    OUT  = BASE / "data/spatial_outputs/video_outputs"

    logger = AlertLogger(
        csv_path       = OUT / "alerts.csv",
        json_path      = OUT / "video_metrics.json",
        debounce_frames= 10,
    )

    print(f"[Test] Simulando 50 frames con debounce_frames=10")
    print(f"{'Frame':<8} {'Level':<8} {'Score':<8} {'Alerta'}")
    print("-" * 40)

    for frame in range(50):
        if frame < 20:
            level, score, unc = "low",    0.50,  None
        elif frame < 35:
            level, score, unc = "medium", 0.862, None
        else:
            level, score, unc = "high",   0.871, 0.023

        alerta = logger.update(
            frame_count  = frame,
            level        = level,
            risk_score   = score,
            uncertainty  = unc,
            timestamp_ms = 15.2 + frame * 0.1,   # latencias sintéticas
        )

        marker = f"⚑  ALERTA {alerta.upper()}" if alerta else ""
        print(f"{frame:<8} {level:<8} {score:<8.3f} {marker}")

    # Persistencia
    logger.save_metrics(video_path="test.mp4", fps_real=12.5)

    # Resumen
    logger.summary()

    # Contenido del CSV
    print(f"\n[CSV] {logger.csv_path}")
    print(logger.csv_path.read_text())

    # Contenido del JSON
    print(f"[JSON] {logger.json_path}")
    print(logger.json_path.read_text())
