python hf_upload.py download-dataset --output-dir tiny_stories # needs to be the one from download_the_data branch download where_data/{split}.parquet
python test_generator.py --dataset-file tiny_stories/validation.parquet --sample 128 #create submission by removing characters from end of test data
python hf_upload.py submit ./sample_submission.csv # submits to huggingface datasets
python llm_scoring.py download-submissions --output-dir downloaded_submissions
python llm_scoring.py evaluate --submissions-dir downloaded_submissions /home/molly/exllamav2/phi-4-exl2 tiny_stories/validation.parquet --sample 128 --max-new-tokens 128 # 128 scores-> partial completion vs full completion 
