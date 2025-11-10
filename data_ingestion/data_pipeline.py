import os
import json
from typing import Dict, List, Any, Optional
from data_ingestion.file_loaders import FileLoader
from data_ingestion.schema_extractor import SchemaExtractor
from data_ingestion.llm_chunker import LLMChunker
from data_ingestion.entity_relation_extractor import EntityRelationExtractor
from data_ingestion.embedding_storage import EmbeddingStorage

class DataIngestionPipeline:
    """Main pipeline that orchestrates the entire data ingestion process"""
    
    def __init__(self, 
                 max_chunk_size: int = 512,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 output_dir: str = "pipeline_output"):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.file_loader = FileLoader()
        self.schema_extractor = SchemaExtractor()
        self.chunker = LLMChunker(max_chunk_size=max_chunk_size)
        self.entity_extractor = EntityRelationExtractor(spacy_model=spacy_model)
        self.embedding_storage = EmbeddingStorage(model_name=embedding_model)
        
        # Pipeline state
        self.pipeline_state = {
            'files_processed': [],
            'schemas': {},
            'chunks': [],
            'knowledge_graph': {},
            'embedding_paths': {}
        }
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file through the entire pipeline"""
        print(f"Processing file: {file_path}")
        
        try:
            # Step 1: Load file
            print("  Step 1: Loading file...")
            data = self.file_loader.load_file(file_path)
            file_info = self.file_loader.get_file_info(file_path)
            file_type = os.path.splitext(file_path)[1].lower()
            
            # Step 2: Extract schema
            print("  Step 2: Extracting schema...")
            schema = self.schema_extractor.extract_schema(data, file_type)
            
            # Step 3: Create chunks
            print("  Step 3: Creating semantic chunks...")
            chunks = self.chunker.chunk_data(data, schema, file_type)
            
            # Update pipeline state
            self.pipeline_state['files_processed'].append(file_path)
            self.pipeline_state['schemas'][file_path] = schema
            self.pipeline_state['chunks'].extend(chunks)
            
            result = {
                'file_path': file_path,
                'file_info': file_info,
                'schema': schema,
                'chunks_created': len(chunks),
                'status': 'success'
            }
            
            print(f"  ✓ Successfully processed {file_path} - Created {len(chunks)} chunks")
            return result
            
        except Exception as e:
            error_result = {
                'file_path': file_path,
                'status': 'error',
                'error': str(e)
            }
            print(f"  ✗ Error processing {file_path}: {str(e)}")
            return error_result
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files through the pipeline"""
        results = []
        
        for file_path in file_paths:
            result = self.process_file(file_path)
            results.append(result)
        
        return results
    
    def create_knowledge_graph(self) -> Dict[str, Any]:
        """Create knowledge graph from all processed chunks"""
        if not self.pipeline_state['chunks']:
            print("No chunks available for knowledge graph creation")
            return {}
        
        print("Creating knowledge graph...")
        print(f"  Processing {len(self.pipeline_state['chunks'])} chunks...")
        
        # Create knowledge graph
        knowledge_graph = self.entity_extractor.create_knowledge_graph(
            self.pipeline_state['chunks']
        )
        
        # Save knowledge graph
        graph_path = os.path.join(self.output_dir, "knowledge_graph.json")
        self.entity_extractor.save_graph_to_json(knowledge_graph, graph_path)
        
        # Get statistics
        graph_stats = self.entity_extractor.get_graph_statistics(knowledge_graph)
        
        # Update pipeline state
        self.pipeline_state['knowledge_graph'] = knowledge_graph
        
        print(f"  ✓ Knowledge graph created with {graph_stats['nodes']} nodes and {graph_stats['edges']} edges")
        
        return {
            'graph_path': graph_path,
            'statistics': graph_stats,
            'graph_data': knowledge_graph
        }
    
    def create_embeddings(self) -> Dict[str, Any]:
        """Create embeddings and FAISS index from all chunks"""
        if not self.pipeline_state['chunks']:
            print("No chunks available for embedding creation")
            return {}
        
        print("Creating embeddings and FAISS index...")
        print(f"  Processing {len(self.pipeline_state['chunks'])} chunks...")
        
        # Create embeddings storage directory
        embeddings_dir = os.path.join(self.output_dir, "embeddings")
        
        # Store embeddings and create FAISS index
        embedding_paths = self.embedding_storage.store_embeddings_and_chunks(
            self.pipeline_state['chunks'], 
            embeddings_dir
        )
        
        # Get storage statistics
        storage_stats = self.embedding_storage.get_storage_statistics(embeddings_dir)
        
        # Update pipeline state
        self.pipeline_state['embedding_paths'] = embedding_paths
        
        print(f"  ✓ Embeddings created and stored in {embeddings_dir}")
        
        return {
            'embeddings_dir': embeddings_dir,
            'file_paths': embedding_paths,
            'statistics': storage_stats
        }
    
    def run_full_pipeline(self, file_paths: List[str]) -> Dict[str, Any]:
        """Run the complete data ingestion pipeline"""
        print("=" * 60)
        print("STARTING FULL DATA INGESTION PIPELINE")
        print("=" * 60)
        
        pipeline_results = {
            'file_processing': [],
            'knowledge_graph': {},
            'embeddings': {},
            'pipeline_summary': {}
        }
        
        try:
            # Step 1: Process all files
            print("\n📁 PHASE 1: FILE PROCESSING")
            print("-" * 40)
            file_results = self.process_multiple_files(file_paths)
            pipeline_results['file_processing'] = file_results
            
            # Step 2: Create knowledge graph
            print("\n🕸️  PHASE 2: KNOWLEDGE GRAPH CREATION")
            print("-" * 40)
            graph_results = self.create_knowledge_graph()
            pipeline_results['knowledge_graph'] = graph_results
            
            # Step 3: Create embeddings
            print("\n🔍 PHASE 3: EMBEDDING CREATION")
            print("-" * 40)
            embedding_results = self.create_embeddings()
            pipeline_results['embeddings'] = embedding_results
            
            # Step 4: Save pipeline summary
            print("\n💾 PHASE 4: SAVING PIPELINE SUMMARY")
            print("-" * 40)
            summary = self.create_pipeline_summary()
            pipeline_results['pipeline_summary'] = summary
            
            print("\n" + "=" * 60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
            return pipeline_results
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            pipeline_results['error'] = str(e)
            return pipeline_results
    
    def create_pipeline_summary(self) -> Dict[str, Any]:
        """Create a summary of the entire pipeline execution"""
        summary = {
            'files_processed': len(self.pipeline_state['files_processed']),
            'total_chunks_created': len(self.pipeline_state['chunks']),
            'schemas_extracted': len(self.pipeline_state['schemas']),
            'knowledge_graph_created': bool(self.pipeline_state['knowledge_graph']),
            'embeddings_created': bool(self.pipeline_state['embedding_paths']),
            'output_directory': self.output_dir,
            'processed_files': self.pipeline_state['files_processed']
        }
        
        # Add detailed statistics if available
        if self.pipeline_state['knowledge_graph']:
            kg_stats = self.entity_extractor.get_graph_statistics(
                self.pipeline_state['knowledge_graph']
            )
            summary['knowledge_graph_stats'] = kg_stats
        
        if self.pipeline_state['embedding_paths']:
            embeddings_dir = os.path.join(self.output_dir, "embeddings")
            if os.path.exists(embeddings_dir):
                storage_stats = self.embedding_storage.get_storage_statistics(embeddings_dir)
                summary['embedding_stats'] = storage_stats
        
        # Save summary to file
        summary_path = os.path.join(self.output_dir, "pipeline_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Pipeline summary saved to {summary_path}")
        
        return summary
    
    def search_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic search"""
        if not self.pipeline_state['embedding_paths']:
            print("Embeddings not available. Please run the pipeline first.")
            return []
        
        # Load embeddings if not already loaded
        embeddings_dir = os.path.join(self.output_dir, "embeddings")
        if not self.embedding_storage.index:
            success = self.embedding_storage.load_embeddings_and_index(embeddings_dir)
            if not success:
                print("Failed to load embeddings for search")
                return []
        
        # Perform search
        results = self.embedding_storage.search_similar_chunks(query, k)
        
        print(f"Found {len(results)} similar chunks for query: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result['similarity_score']:.3f} - {result['metadata']['content_preview']}")
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        return {
            'files_processed': len(self.pipeline_state['files_processed']),
            'chunks_created': len(self.pipeline_state['chunks']),
            'knowledge_graph_ready': bool(self.pipeline_state['knowledge_graph']),
            'embeddings_ready': bool(self.pipeline_state['embedding_paths']),
            'output_directory': self.output_dir,
            'last_processed_files': self.pipeline_state['files_processed'][-5:] if self.pipeline_state['files_processed'] else []
        }