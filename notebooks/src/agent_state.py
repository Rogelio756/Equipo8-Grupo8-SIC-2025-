# =============================================================
# src/agent_state.py
# Estado del agente de percepción-acción para video BDD100K
# Equipo8-Grupo8-SIC-2025
# =============================================================

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import time


class AgentState:
    """
    Estado centralizado del agente de percepción-acción.

    Gestiona el nivel de riesgo actual con histéresis para evitar
    oscilaciones rápidas, y aplica un strategy pattern para
    determinar qué modelos activar según el nivel.
    """

    def __init__(self):
        self.frame_buffer: deque        = deque(maxlen=10)
        self.risk_history: deque        = deque(maxlen=30)
        self.current_level: str         = "low"
        self.frame_count: int           = 0
        self._frames_bajo_umbral: int   = 0

        # Hiperparámetros de histéresis y umbrales BNN
        self.HYSTERESIS_DOWN: int       = 5
        self.THRESH_MEDIUM: float       = 0.62     # p50 del rango real BNN en video BDDA
        self.THRESH_HIGH: float         = 0.68     # p80 del rango real BNN en video BDDA

    # ----------------------------------------------------------
    # Actualización de estado
    # ----------------------------------------------------------

    def update(self, frame, risk_score: float) -> str:
        """
        Agrega frame al buffer, registra el score, actualiza el nivel
        y retorna el nivel actual.
        """
        self.frame_buffer.append(frame)
        self.risk_history.append(risk_score)
        self._update_level(risk_score)
        self.frame_count += 1
        return self.current_level

    def _update_level(self, score: float) -> None:
        """
        Actualiza current_level con histéresis:
        - Subida inmediata al superar umbrales.
        - Bajada lenta: requiere HYSTERESIS_DOWN frames consecutivos bajo umbral.
        """
        # Subida inmediata a "high"
        if score >= self.THRESH_HIGH:
            self.current_level = "high"
            self._frames_bajo_umbral = 0
            return

        # Subida inmediata a "medium" (solo desde "low")
        if score >= self.THRESH_MEDIUM and self.current_level == "low":
            self.current_level = "medium"
            self._frames_bajo_umbral = 0
            return

        # Bajada lenta
        if score < self.THRESH_MEDIUM:
            self._frames_bajo_umbral += 1
            if self._frames_bajo_umbral >= self.HYSTERESIS_DOWN:
                if self.current_level == "high":
                    self.current_level = "medium"
                elif self.current_level == "medium":
                    self.current_level = "low"
                self._frames_bajo_umbral = 0
        else:
            # score >= THRESH_MEDIUM pero ya estamos en "medium" o "high" → reset
            self._frames_bajo_umbral = 0

    # ----------------------------------------------------------
    # Strategy pattern — selección de modelos
    # ----------------------------------------------------------

    def active_models(self) -> List[str]:
        """Retorna los modelos a activar según el nivel de riesgo actual."""
        if self.current_level == "low":
            return ["yolo"]
        elif self.current_level == "medium":
            return ["yolo", "segformer"]
        else:  # "high"
            return ["yolo", "segformer", "resnet"]

    # ----------------------------------------------------------
    # Inspección
    # ----------------------------------------------------------

    def status(self) -> None:
        """Imprime el estado actual del agente."""
        ultimo_score  = self.risk_history[-1] if self.risk_history else float("nan")
        avg_score     = (sum(self.risk_history) / len(self.risk_history)
                         if self.risk_history else float("nan"))
        modelos       = self.active_models()

        print(f"\n[AgentState] Estado actual")
        print(f"  nivel         : {self.current_level}")
        print(f"  frames proc.  : {self.frame_count}")
        print(f"  último score  : {ultimo_score:.4f}")
        print(f"  promedio score: {avg_score:.4f}")
        print(f"  modelos activos: {modelos}")


# =============================================================
# Bloque de prueba — ejecutar directamente para verificar
# =============================================================

if __name__ == "__main__":
    state = AgentState()

    escenarios = [
        (range(0,  5),  0.50),   # debe quedarse en "low"
        (range(5,  10), 0.862),  # debe subir a "medium" inmediato
        (range(10, 15), 0.870),  # debe subir a "high" inmediato
        (range(15, 20), 0.50),   # debe bajar después de 5 frames consecutivos
    ]

    print(f"{'Frame':<8} {'Score':<8} {'Nivel':<10} Modelos activos")
    print("-" * 55)

    for rango, score in escenarios:
        for i in rango:
            nivel = state.update(frame=None, risk_score=score)
            modelos = state.active_models()
            print(f"{i:<8} {score:<8.3f} {nivel:<10} {modelos}")

    state.status()
