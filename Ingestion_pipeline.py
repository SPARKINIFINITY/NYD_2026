from data_ingestion.data_pipeline import DataIngestionPipeline  # Correct import
# Import the pipeline class

# Define the file path to the dataset
dataset_file_path = r"D:\Nyd_2026\uploaded_files\bhagavad_gita_dataset.json"  # Update with the actual path of your dataset

# Create an instance of the pipeline
pipeline = DataIngestionPipeline(output_dir="pipeline_output")  # You can change the output directory if you want

# Run the full pipeline with the dataset file path
result = pipeline.run_full_pipeline([dataset_file_path])

# Print the results to see the output of each phase
print("=" * 60)
print("Pipeline Results:")
print("=" * 60)

# Display the results from the pipeline
print(f"Files processed: {result['file_processing']}")
print(f"Knowledge Graph: {result['knowledge_graph']}")
print(f"Embeddings: {result['embeddings']}")
print(f"Pipeline Summary: {result['pipeline_summary']}")

# Optionally, print the entire result if needed
#print(result)
