import React from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface FilteringStep {
  stage: string;
  count: number;
  description: string;
  label: string;
}

interface FilteringGraphProps {
  messageId: string;
  filteringSteps?: FilteringStep[];
}

const FilteringGraph: React.FC<FilteringGraphProps> = ({ messageId, filteringSteps = [] }) => {
  // If no filtering steps provided, show a placeholder
  if (!filteringSteps || filteringSteps.length === 0) {
    return (
      <div className="w-full h-80 bg-gradient-to-br from-background via-muted/30 to-background rounded-xl border shadow-lg flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <div className="text-sm font-medium">No filtering data available</div>
          <div className="text-xs mt-1">Filtering steps will appear here when available</div>
          </div>
          </div>
    );
  }

  // Generate nodes dynamically from filtering steps
  const nodes: Node[] = filteringSteps.map((step, index) => {
    const isFirst = index === 0;
    const isLast = index === filteringSteps.length - 1;
    
    // Calculate position based on step index
    const x = 160; // Center horizontally
    const y = index * 80; // Stack vertically with spacing
    
    return {
      id: step.stage,
      type: isFirst ? 'input' : isLast ? 'output' : 'default',
      data: { 
        label: (
          <div className="text-center">
            <div className={`font-medium text-sm ${isFirst || isLast ? 'font-semibold' : ''}`}>
              {step.label}
          </div>
            <div className="text-xs opacity-80 mt-1">
              {step.count} {step.count === 1 ? 'item' : 'items'}
          </div>
          </div>
        )
      },
      position: { x, y },
      style: { 
        background: isFirst || isLast 
          ? 'hsl(var(--primary))' 
          : 'hsl(var(--secondary))',
        color: isFirst || isLast 
          ? 'hsl(var(--primary-foreground))' 
          : 'hsl(var(--secondary-foreground))',
        border: isFirst || isLast 
          ? 'none' 
          : '1px solid hsl(var(--border))',
        borderRadius: isFirst || isLast ? '12px' : '10px',
        width: 180,
        height: isFirst || isLast ? 60 : 55,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: isFirst || isLast 
          ? '0 4px 12px hsl(var(--primary) / 0.25)' 
          : '0 2px 8px hsl(var(--muted) / 0.2)'
      }
    };
  });

  // Generate edges connecting consecutive nodes
  const edges: Edge[] = [];
  for (let i = 0; i < filteringSteps.length - 1; i++) {
    const currentStep = filteringSteps[i];
    const nextStep = filteringSteps[i + 1];
    
    edges.push({
      id: `e-${currentStep.stage}-${nextStep.stage}`,
      source: currentStep.stage,
      target: nextStep.stage,
      type: 'smoothstep',
      animated: true,
      style: { 
        stroke: 'hsl(var(--primary))', 
        strokeWidth: i === 0 ? 2.5 : 2 
      }
    });
  }

  return (
    <div className="w-full h-80 bg-gradient-to-br from-background via-muted/30 to-background rounded-xl border shadow-lg overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        attributionPosition="bottom-right"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        className="bg-transparent"
      >
        <Background 
          color="hsl(var(--border))" 
          gap={24} 
          size={1}
        />
        <MiniMap 
          nodeColor="hsl(var(--primary))"
          nodeStrokeWidth={2}
          pannable={false}
          zoomable={false}
          style={{
            height: 90,
            width: 140,
            backgroundColor: 'hsl(var(--background) / 0.95)',
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px',
          }}
        />
      </ReactFlow>
    </div>
  );
};

export default FilteringGraph;