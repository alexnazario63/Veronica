import React, { useEffect, useState } from "react";
import type {
  ArchitectureNode,
  StructuredAssistantPayload,
  TrainingStepMetric,
} from "./types/veronica";

export const VeronicaApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"chat" | "observatory" | "lab" | "noc">("chat");
  const [userId, setUserId] = useState<string>("usr-eng-01");
  const [nodes, setNodes] = useState<ArchitectureNode[]>([]);
  const [metrics, setMetrics] = useState<TrainingStepMetric[]>([]);
  const [lastPayload, setLastPayload] = useState<StructuredAssistantPayload | null>(null);

  useEffect(() => {
    fetch("/api/neural/architecture")
      .then((r) => r.json())
      .then((d) => setNodes(d.nodes || []));
    fetch("/api/neural/metrics")
      .then((r) => r.json())
      .then((d) => setMetrics(d.history || []));
  }, []);

  return (
    <div className="veronica-shell" data-user={userId} data-tab={activeTab}>
      <header>
        <h1>VERONICA — Net Neural AI</h1>
        <span>Camadas Neurais: {nodes.length} | Steps: {metrics.length}</span>
        {lastPayload && <span>Confiança: {Math.round(lastPayload.confidence * 100)}%</span>}
      </header>
    </div>
  );
};

export default VeronicaApp;
