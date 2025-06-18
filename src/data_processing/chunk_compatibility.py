import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityChunker:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.chunks: List[Dict[str, Any]] = []
        
    def load_compatibility_data(self, filepath: str = 'data/processed/compatibility_analysis.json') -> Dict:
        """Load the compatibility analysis data."""
        logger.info(f"Loading compatibility data from {filepath}")
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def chunk_server_data(self, servers: List[Dict]) -> List[Dict]:
        """Chunk server data into smaller pieces."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for server in servers:
            # Convert server to string representation for size estimation
            server_str = json.dumps(server)
            server_size = len(server_str)
            
            if current_size + server_size > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append({
                    "type": "server_chunk",
                    "servers": current_chunk,
                    "chunk_size": current_size
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(server)
            current_size += server_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "type": "server_chunk",
                "servers": current_chunk,
                "chunk_size": current_size
            })
        
        return chunks
    
    def chunk_summary_data(self, summaries: Dict) -> List[Dict]:
        """Chunk summary data (environment and manufacturer summaries)."""
        chunks = []
        
        # Chunk environment summaries
        for env, summary in summaries.get("environment_summaries", {}).items():
            chunk = {
                "type": "environment_summary",
                "environment": env,
                "data": summary
            }
            chunks.append(chunk)
        
        # Chunk manufacturer summaries
        for mfr, summary in summaries.get("manufacturer_summaries", {}).items():
            chunk = {
                "type": "manufacturer_summary",
                "manufacturer": mfr,
                "data": summary
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_metadata(self, metadata: Dict) -> Dict:
        """Create a metadata chunk."""
        return {
            "type": "metadata",
            "data": metadata
        }
    
    def process_data(self, data: Dict) -> List[Dict]:
        """Process the entire compatibility data into chunks."""
        logger.info("Processing compatibility data into chunks")
        
        # Add metadata chunk
        self.chunks.append(self.chunk_metadata(data["metadata"]))
        
        # Process server data
        server_chunks = self.chunk_server_data(data["servers"])
        self.chunks.extend(server_chunks)
        
        # Process summary data
        summary_chunks = self.chunk_summary_data({
            "environment_summaries": data["environment_summaries"],
            "manufacturer_summaries": data["manufacturer_summaries"]
        })
        self.chunks.extend(summary_chunks)
        
        return self.chunks
    
    def save_chunks(self, output_dir: str = 'data/processed/chunks'):
        """Save chunks to individual files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, chunk in enumerate(self.chunks):
            chunk_file = output_path / f"chunk_{i:04d}.json"
            with open(chunk_file, 'w') as f:
                json.dump(chunk, f, indent=2)
        
        logger.info(f"Saved {len(self.chunks)} chunks to {output_dir}")

def main():
    chunker = CompatibilityChunker()
    
    # Load and process data
    data = chunker.load_compatibility_data()
    chunks = chunker.process_data(data)
    
    # Save chunks
    chunker.save_chunks()
    
    # Log summary
    chunk_types = {}
    for chunk in chunks:
        chunk_type = chunk["type"]
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    logger.info("Chunk processing complete")
    logger.info("Chunk type distribution:")
    for chunk_type, count in chunk_types.items():
        logger.info(f"{chunk_type}: {count} chunks")

if __name__ == "__main__":
    main() 