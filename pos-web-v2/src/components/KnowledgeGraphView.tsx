import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import type { KnowledgeGraph, GraphNode, GraphLink, Entity, Relation } from '@types/index';
import { ZoomInIcon, ZoomOutIcon, ArrowsPointingOutIcon, AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline';

interface KnowledgeGraphViewProps {
  graph: KnowledgeGraph;
  selectedNodeId?: string | null;
  onNodeClick?: (node: GraphNode) => void;
  onNodeDoubleClick?: (node: GraphNode) => void;
  className?: string;
  width?: number;
  height?: number;
}

export const KnowledgeGraphView: React.FC<KnowledgeGraphViewProps> = ({
  graph,
  selectedNodeId,
  onNodeClick,
  onNodeDoubleClick,
  className = '',
  width = 800,
  height = 600,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showLabels, setShowLabels] = useState(true);
  const [nodeSizeByConnections, setNodeSizeByConnections] = useState(true);
  
  const colors = [
    '#3b82f6', // blue
    '#22c55e', // green
    '#eab308', // yellow
    '#ef4444', // red
    '#a855f7', // purple
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#f97316', // orange
  ];
  
  const getNodeColor = (node: GraphNode) => {
    return colors[node.group % colors.length];
  };
  
  const getNodeRadius = useCallback((node: GraphNode) => {
    if (!nodeSizeByConnections) return 8;
    const connectionCount = graph.links.filter(
      (l) => l.source === node.id || l.target === node.id
    ).length;
    return Math.max(6, Math.min(20, 6 + connectionCount * 2));
  }, [graph.links, nodeSizeByConnections]);
  
  useEffect(() => {
    if (!svgRef.current || graph.nodes.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // 设置SVG
    svg.attr('viewBox', [0, 0, width, height]);
    
    // 创建组
    const g = svg.append('g');
    
    // 缩放行为
    const zoomBehavior = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoom(event.transform.k);
      });
    
    svg.call(zoomBehavior);
    
    // 力导向模拟
    const simulation = d3.forceSimulation<GraphNode>(graph.nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(graph.links)
        .id((d) => d.id)
        .distance(100)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d) => getNodeRadius(d as GraphNode) + 5));
    
    simulationRef.current = simulation;
    
    // 绘制连线
    const link = g.append('g')
      .attr('stroke', '#94a3b8')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(graph.links)
      .join('line')
      .attr('stroke-width', (d) => Math.sqrt(d.value || 1));
    
    // 绘制节点
    const node = g.append('g')
      .selectAll('g')
      .data(graph.nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<SVGGElement, GraphNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );
    
    // 节点圆圈
    node.append('circle')
      .attr('r', (d) => getNodeRadius(d))
      .attr('fill', (d) => getNodeColor(d))
      .attr('stroke', (d) => d.id === selectedNodeId ? '#fff' : 'transparent')
      .attr('stroke-width', (d) => d.id === selectedNodeId ? 4 : 0)
      .attr('class', 'transition-all duration-200');
    
    // 节点标签
    node.append('text')
      .text((d) => d.name)
      .attr('x', (d) => getNodeRadius(d) + 5)
      .attr('y', 4)
      .attr('font-size', '12px')
      .attr('fill', '#374151')
      .attr('font-family', 'Inter, sans-serif')
      .style('pointer-events', 'none')
      .style('opacity', showLabels ? 1 : 0)
      .style('transition', 'opacity 0.2s');
    
    // 节点交互
    node.on('click', (event, d) => {
      event.stopPropagation();
      onNodeClick?.(d);
    });
    
    node.on('dblclick', (event, d) => {
      event.stopPropagation();
      onNodeDoubleClick?.(d);
    });
    
    // 悬浮提示
    node.append('title')
      .text((d) => `${d.name}\n类型: ${d.type}\n连接数: ${graph.links.filter(l => l.source === d.id || l.target === d.id).length}`);
    
    // 更新位置
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      
      node.attr('transform', (d) => `translate(${d.x},${d.y})`);
    });
    
    return () => {
      simulation.stop();
    };
  }, [graph, width, height, selectedNodeId, showLabels, nodeSizeByConnections, onNodeClick, onNodeDoubleClick, getNodeRadius]);
  
  // 控制函数
  const handleZoomIn = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .call(d3.zoom().transform, d3.zoomTransform(svgRef.current).scale(zoom * 1.2));
    }
  };
  
  const handleZoomOut = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .call(d3.zoom().transform, d3.zoomTransform(svgRef.current).scale(zoom * 0.8));
    }
  };
  
  const handleReset = () => {
    if (svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .call(d3.zoom().transform, d3.zoomIdentity);
      simulationRef.current?.alpha(1).restart();
    }
  };
  
  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* 控制面板 */}
      <div className="absolute top-4 left-4 z-10 bg-white dark:bg-dark-800 rounded-lg shadow-lg p-3 space-y-3">
        <div className="flex flex-col gap-1">
          <button
            onClick={handleZoomIn}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 transition-colors"
            title="放大"
          >
            <ZoomInIcon className="w-5 h-5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 transition-colors"
            title="缩小"
          >
            <ZoomOutIcon className="w-5 h-5" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 transition-colors"
            title="重置视图"
          >
            <ArrowsPointingOutIcon className="w-5 h-5" />
          </button>
        </div>
        
        <hr className="border-gray-200 dark:border-dark-700" />
        
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              className="rounded"
            />
            显示标签
          </label>
          
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={nodeSizeByConnections}
              onChange={(e) => setNodeSizeByConnections(e.target.checked)}
              className="rounded"
            />
            按连接数调整大小
          </label>
        </div>
      </div>
      
      {/* 统计信息 */}
      <div className="absolute top-4 right-4 z-10 bg-white dark:bg-dark-800 rounded-lg shadow-lg p-3">
        <div className="text-sm space-y-1">
          <p>
            <span className="text-gray-500">节点: </span>
            <span className="font-medium">{graph.nodes.length}</span>
          </p>
          <p>
            <span className="text-gray-500">关系: </span>
            <span className="font-medium">{graph.links.length}</span>
          </p>
          <p>
            <span className="text-gray-500">缩放: </span>
            <span className="font-medium">{Math.round(zoom * 100)}%</span>
          </p>
        </div>
      </div>
      
      {/* SVG容器 */}
      <svg
        ref={svgRef}
        className="w-full h-full bg-gray-50 dark:bg-dark-900 rounded-lg"
        style={{ minHeight: '400px' }}
      />
    </div>
  );
};

export default KnowledgeGraphView;
