import os
import json
import pandas as pd

def folder_to_dataframe(folder_path):
    """
    Reads all JSON files from a specified folder and converts them into a pandas DataFrame,
    including the source filename for each record.
    
    Args:
    folder_path (str): The path to the folder containing JSON files.
    
    Returns:
    pd.DataFrame: A DataFrame containing the combined data from all JSON files.
    """
    data_list = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item['filename'] = filename
                        data_list.extend(data)
                    elif isinstance(data, dict):
                        data['filename'] = filename
                        data_list.append(data)
                        
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {filename}. Skipping.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    df = pd.DataFrame(data_list)
    return df

def clean_format_dataframe(df):
    """
    Formats the DataFrame by ensuring consistent column names, data types,
    and one-hot encoding the tags column.
    """
    columns_to_keep = ['tags', 'source_code', 'difficulty','prob_desc_input_spec','prob_desc_output_spec','prob_desc_description','prob_desc_notes']
    df_cleaned = df[columns_to_keep].copy()
    
    distinct_tuples = df_cleaned['tags'].apply(tuple).unique()
    distinct_lists = [list(x) for x in distinct_tuples]
    print(f"Distinct tag combinations found: {len(distinct_lists)}")
    
    tags_to_keep = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
    
    def filter_tags(tag_list):
        return [tag for tag in tag_list if tag in tags_to_keep]
    
    df_cleaned['tags'] = df_cleaned['tags'].apply(filter_tags)

    for tag in tags_to_keep:
        col_name = f"tag_{tag.replace(' ', '_')}" 
        df_cleaned[col_name] = df_cleaned['tags'].apply(lambda x: 1 if tag in x else 0)
        
    return df_cleaned

if __name__ == "__main__":
    folder_path = '../code_classification_dataset'
    df = folder_to_dataframe(folder_path)
    df_processed = clean_format_dataframe(df)
    tag_columns = [col for col in df_processed.columns if col.startswith('tag_')]
    
    stats_list = []
    
    for col in tag_columns:
        # Count Positives (1) and Negatives (0)
        pos_count = df_processed[col].sum()
        neg_count = len(df_processed) - pos_count
        pos_ratio = pos_count / len(df_processed)
        
        stats_list.append({
            'Tag': col,
            'Positive (1)': pos_count,
            'Negative (0)': neg_count,
            'Pos Ratio': f"{pos_ratio:.1%}"
        })

    # Create a nice summary dataframe for display
    stats_df = pd.DataFrame(stats_list)
    
    # Sort by number of positive samples (descending)
    stats_df = stats_df.sort_values(by='Positive (1)', ascending=False)
    
    print(stats_df.to_string(index=False))
    print("-" * 40)
    print(f"Total Samples: {len(df_processed)}")
    