import os
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

# 設定 PyTorch 記憶體擴展
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 動態調整向量長度
def match_length(vector, target_length):
    current_length = vector.size(0)
    if current_length > target_length:
        return vector[:target_length]
    elif current_length < target_length:
        return torch.nn.functional.pad(vector, (0, target_length - current_length), mode='constant')
    return vector

# 加載 .prottrans 資料
def load_prottrans(file_path):
    embeddings = []
    with open(file_path, "r") as f:
        for line in f:
            embeddings.append(list(map(float, line.strip().split())))
    return torch.tensor(embeddings, dtype=torch.float16, device=device)

# 加載資料庫文件夾中的所有 .prottrans 文件
def load_database(folder_path):
    all_embeddings = []
    for file_name in tqdm(os.listdir(folder_path), desc="Loading database files", unit="file"):
        if file_name.endswith(".esm"):
            file_path = os.path.join(folder_path, file_name)
            embeddings = load_prottrans(file_path)
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)

# 保存資料
def save_prottrans(path, data):
    data = data.cpu().numpy()
    with open(path, "w") as f:
        for embedding in data:
            f.write(" ".join(map(str, embedding)) + "\n")
    print(f"Data saved to {path}")

# 向量化的距離計算（啟用分批處理）
def vector_distance(target_embedding, database, batch_size=32):  # 減小批次大小
    target_len = target_embedding.size(1)
    database_adjusted = torch.stack([match_length(vec, target_len) for vec in database])

    processed_embeddings = []
    for i in tqdm(range(0, target_embedding.size(0), batch_size), desc="Processing embeddings in batches", unit="batch"):
        batch = target_embedding[i:i+batch_size]
        distances = torch.cdist(batch, database_adjusted)
        nearest_indices = torch.argsort(distances, dim=1)[:, :5]
        nearest_vectors = database_adjusted[nearest_indices]
        avg_embedding = torch.mean(nearest_vectors, dim=1)
        combined_embedding = (batch + avg_embedding) / 2
        processed_embeddings.append(combined_embedding)
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()

    return torch.cat(processed_embeddings, dim=0)

# 並行處理單個文件
def process_file(input_path, output_path, database):
    try:
        target_embedding = load_prottrans(input_path)
        processed_embedding = vector_distance(target_embedding, database)
        save_prottrans(output_path, processed_embedding)
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

# 主程序
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--path_input", type=str, required=True, help="Input folder containing .prottrans files (L1 files)")
    parser.add_argument("-out", "--path_output", type=str, required=True, help="Output folder for processed L1 files")
    parser.add_argument("-db", "--database_folder", type=str, required=True, help="Database folder containing multiple .prottrans files")
    parser.add_argument("-jobs", "--num_jobs", type=int, default=1, help="Number of parallel jobs (default: use all cores)")
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        print(f"Error: Input folder {args.path_input} does not exist!")
        exit(1)

    print("Loading L2 database embeddings from folder...")
    original_database = load_database(args.database_folder)
    database = original_database.clone()

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    input_files = [
        os.path.join(args.path_input, file_name)
        for file_name in os.listdir(args.path_input)
        if file_name.endswith(".esm")
    ]
    output_files = [
        os.path.join(args.path_output, os.path.basename(file_name))
        for file_name in input_files
    ]

    print(f"Processing {len(input_files)} files in parallel...")
    Parallel(n_jobs=args.num_jobs)(
        delayed(process_file)(in_file, out_file, database)
        for in_file, out_file in zip(input_files, output_files)
    )

    if torch.equal(original_database, database):
        print("Database integrity verified. No modifications were made.")
    else:
        print("Warning: Database was modified!")
