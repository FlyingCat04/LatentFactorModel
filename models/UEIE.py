from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class LatentFactorModel(nn.Module):
    def __init__(
        self,
        db_config=None, # THAY ĐỔI: Nhận db_config thay vì connection
        ratings=None, 
        items=None,
        k=90,
        weight=0.3,
        lam=0.001,
        lr=0.01,
        model_id=1,
        domain_id=None,
        train_mode='train',
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.db_config = db_config # THAY ĐỔI
        self.weight = weight
        self.lam = lam
        self.lr = lr
        self.model_id = model_id
        self.train_mode = train_mode.lower()
        self.k = k
        self.tokenizer = None
        self.bert_model = None
        self.ratings = [] 
        self.test_ratings = []
        self.user_emb_dict = {}
        self.item_emb_dict = {}
        self.users = []
        self.items_list = []
        self.user2idx = {}
        self.item2idx = {}
        self.mu = 3.5
        self.model = None
        self.optimizer = None
        self.inferred_prefs = None
        self.domain_id = domain_id

        if self.db_config is None and self.train_mode == 'train' and (ratings is None or items is None):
            raise ValueError("❌ DB config or ratings/items required for training.")
        if self.db_config is None and self.train_mode == 'load':
            raise ValueError("❌ DB config required for loading model.")
        if self.train_mode not in ['train', 'load']:
            raise ValueError("❌ train_mode must be 'train' or 'load'")

        if self.train_mode == 'load':
            print("--- Loading Existing Model ---")
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            self.load_model_from_db(model_id=model_id)
            if self.model is None or not self.users or not self.items_list:
                print("⚠️ Warning: Model load incomplete (maybe cold start). Continuing...")

        else:
            if self.k is None:
                raise ValueError("❌ Parameter 'k' (e.g., 90) is required for training.")

            if ratings is None:
                self.load_ratings_from_db(ratio=1)
            else:
                self.ratings, self.test_ratings = ratings, []

            if not self.ratings:
                print("⚠️ Warning: No training ratings available. Model will initialize with zeros/random.")
                users_in_train = []
                items_in_train = []
            else:
                users_in_train_set = set(u for u, _, _ in self.ratings)
                items_in_train_set = set(i for _, i, _ in self.ratings)
                users_in_train = list(users_in_train_set)
                items_in_train = list(items_in_train_set)

            # if users_in_train and items_in_train:
            #     self.load_user_item_embeddings_from_db(users_in_train, items_in_train)
            
            self.load_user_item_from_db()
            self.user2idx = {u: i for i, u in enumerate(self.users)}
            self.item2idx = {i: j for j, i in enumerate(self.items_list)}
            
            if self.users and self.items_list:
                self.load_user_item_embeddings_from_db(self.users, self.items_list)
            
            if self.ratings:
                original_count = len(self.ratings)
                self.ratings = [(u, i, r) for u, i, r in self.ratings if u in self.user2idx and i in self.item2idx]
                filtered_count = original_count - len(self.ratings)
                if filtered_count > 0:
                    print(f"⚠️ Filtered out {filtered_count} ratings with invalid user/item IDs")
            
            self.compute_user_embeddings()

            self.mu = np.mean([r for _, _, r in self.ratings]) if self.ratings else 3.5

            n_users_final = len(self.users)
            n_items_final = len(self.items_list)
            
            self.model = self.UEIEModel(n_users_final, n_items_final, self.k, self.mu).to(self.device)

            self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=self.lr, 
                                   weight_decay=self.lam)
            
            self.inferred_prefs = self.compute_inferred_preferences()
            print("✅ Initialization complete and ready.")

    class UEIEModel(nn.Module):
        def __init__(self, n_users, n_items, k, mu, P_init=None, Q_init=None, b_u_init=None, b_i_init=None):
            super().__init__()
            n_users = max(n_users, 1)
            n_items = max(n_items, 1)

            self.P = nn.Embedding(n_users, k)
            self.Q = nn.Embedding(n_items, k)
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            self.mu = mu
            
            with torch.no_grad():
                if P_init is not None:
                    t = torch.tensor(P_init, dtype=torch.float32)
                    if t.shape == self.P.weight.shape:
                        self.P.weight.copy_(t)
                    else:
                        nn.init.xavier_uniform_(self.P.weight)
                else:
                    nn.init.xavier_uniform_(self.P.weight)

                if Q_init is not None:
                    t = torch.tensor(Q_init, dtype=torch.float32)
                    if t.shape == self.Q.weight.shape:
                        self.Q.weight.copy_(t)
                    else:
                        nn.init.xavier_uniform_(self.Q.weight)
                else:
                    nn.init.xavier_uniform_(self.Q.weight)

                if b_u_init is not None:
                    t = torch.tensor(b_u_init, dtype=torch.float32).unsqueeze(1)
                    if t.shape == self.b_u.weight.shape:
                        self.b_u.weight.copy_(t)
                    else:
                        nn.init.zeros_(self.b_u.weight)
                else:
                    nn.init.zeros_(self.b_u.weight)

                if b_i_init is not None:
                    t = torch.tensor(b_i_init, dtype=torch.float32).unsqueeze(1)
                    if t.shape == self.b_i.weight.shape:
                        self.b_i.weight.copy_(t)
                    else:
                        nn.init.zeros_(self.b_i.weight)
                else:
                    nn.init.zeros_(self.b_i.weight)

        def forward(self, user_idx, item_idx):
            p_u = self.P(user_idx)
            q_i = self.Q(item_idx)
            b_u = self.b_u(user_idx).squeeze(dim=-1)
            b_i = self.b_i(item_idx).squeeze(dim=-1)
            dot_product = (p_u * q_i).sum(dim=1)
            return self.mu + b_u + b_i + dot_product

    def compute_user_embeddings(self):
        if not self.item_emb_dict:
            k_dim_bert = self.k
            for user_id in self.users:
                if user_id not in self.user_emb_dict:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
            return

        k_dim_bert = 768
        if self.item_emb_dict:
             first_val = next(iter(self.item_emb_dict.values()))
             k_dim_bert = len(first_val)

        user_ratings_map = defaultdict(list)
        for u, i, r in self.ratings:
            if u in self.user2idx:
                user_ratings_map[u].append((i, r))

        for user_id in self.users:
            ratings_list = user_ratings_map.get(user_id, [])
            if not ratings_list:
                if user_id not in self.user_emb_dict or self.user_emb_dict[user_id] is None:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                continue

            user_avg_rating = np.mean([r for _, r in ratings_list])
            tau = user_avg_rating

            liked_item_embeddings = []
            for item_id, rating in ratings_list:
                if rating >= tau:
                    if item_id in self.item_emb_dict and np.any(self.item_emb_dict[item_id] != 0):
                        liked_item_embeddings.append(self.item_emb_dict[item_id])

            if liked_item_embeddings:
                t_u = np.mean(liked_item_embeddings, axis=0)
                if t_u.shape[0] != k_dim_bert:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                else:
                    self.user_emb_dict[user_id] = t_u
            else:
                all_rated_items = [self.item_emb_dict[i] for i, _ in ratings_list if i in self.item_emb_dict and np.any(self.item_emb_dict[i] != 0)]
                if all_rated_items:
                    self.user_emb_dict[user_id] = np.mean(all_rated_items, axis=0)
                else:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_bert)
                
    # THAY ĐỔI: Sử dụng with psycopg2.connect
    def load_user_item_from_db(self):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT "Id" From "User" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.users = [str(row[0]) for row in cur.fetchall()]
                    
                    cur.execute('SELECT "Id" From "Item" WHERE "DomainId" = %s ORDER BY "Id"', (self.domain_id,))
                    self.items_list = [str(row[0]) for row in cur.fetchall()]
        except Exception as e:
            print(f"⚠️ Error loading users/items: {e}")
            self.users = []
            self.items_list = []
        
    # THAY ĐỔI: Sử dụng with psycopg2.connect
    def load_ratings_from_db(self, limit_total=1000000, ratio=0.8):
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "UserId", "ItemId", "Value"
                        FROM "Rating"
                        WHERE "DomainId" = %s AND "Value" IS NOT NULL
                        LIMIT %s
                    """, (self.domain_id, limit_total))
                    rows = cur.fetchall()
        except Exception as e:
            print(f"❌ Error querying ratings: {e}")
            rows = []

        if not rows:
            print("⚠️ No rating data loaded from DB (Cold start).")
            self.ratings = []
            self.test_ratings = []
            return

        train_count = int(len(rows) * ratio)
        if train_count == 0 and len(rows) > 0: train_count = len(rows)

        self.ratings = [(str(u), str(i), float(r)) for u, i, r in rows[:train_count]]
        self.test_ratings = [(str(u), str(i), float(r)) for u, i, r in rows[train_count:]]

    # THAY ĐỔI: Mở kết nối để query items, tắt, rồi mở lại query users (Tránh giữ kết nối lâu lúc BERT chạy)
    def load_user_item_embeddings_from_db(self, users_to_load, items_to_load):
        if not users_to_load or not items_to_load:
            return

        CHUNK_SIZE = 1000
        k_bert_dim = 768
        all_rows_items = []
        items_list_for_query = list(items_to_load)
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(items_list_for_query), CHUNK_SIZE):
                        chunk = items_list_for_query[i : i + CHUNK_SIZE]
                        if not chunk: continue
                        cur.execute("""
                            SELECT "Id", 
                                   COALESCE(NULLIF("Description", ''), "Title") AS "Description", 
                                   "EmbeddingVector"
                            FROM "Item"
                            WHERE "DomainId" = %s AND "Id" = ANY(%s::int[])
                        """, (self.domain_id, chunk,))
                        all_rows_items.extend(cur.fetchall())
        except Exception as e:
            print(f"❌ Error querying item chunks: {e}")
            raise

        self.item_emb_dict = {}
        missing_items_info = []
        processed_item_ids = set()
        for item_id, desc, emb_vec in all_rows_items:
            item_id = str(item_id)
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    if len(loaded_emb) > 0 and np.any(loaded_emb != 0):
                        self.item_emb_dict[item_id] = loaded_emb
                        k_bert_dim = len(loaded_emb) 
                    else:
                        missing_items_info.append((item_id, desc))
                except Exception as e:
                    missing_items_info.append((item_id, desc))
            else:
                if desc is not None or desc != "":
                    missing_items_info.append((item_id, desc))
            processed_item_ids.add(item_id)

        items_not_in_db_table = set(items_to_load) - processed_item_ids
        if items_not_in_db_table:
            for item_id in items_not_in_db_table:
                missing_items_info.append((item_id, None))

        # --- Giai đoạn này dùng BERT, KHÔNG CẦN KẾT NỐI DB ---
        if missing_items_info:
            if self.tokenizer is None or self.bert_model is None:
                try:
                    self.tokenizer, self.bert_model = self.load_bert()
                    if hasattr(self.bert_model, 'config') and hasattr(self.bert_model.config, 'dim'):
                        k_bert_dim = self.bert_model.config.dim
                except Exception as e:
                    print(f"⚠️ Could not load BERT for missing items. Using zeros.")
            
            for item_id, desc in tqdm(missing_items_info, desc="   Computing BERT/Zeros"):
                if item_id not in self.item_emb_dict:
                    if self.bert_model:
                        bert_emb = self.get_embedding(desc)
                        if bert_emb.shape[0] != k_bert_dim:
                            bert_emb = np.zeros(k_bert_dim)
                        self.item_emb_dict[item_id] = bert_emb
                    else:
                        self.item_emb_dict[item_id] = np.zeros(k_bert_dim)
        
        all_rows_users = []
        users_list_for_query = list(users_to_load)
        
        # Mở lại kết nối DB để query User 
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    for i in range(0, len(users_list_for_query), CHUNK_SIZE):
                        chunk = users_list_for_query[i : i + CHUNK_SIZE]
                        if not chunk: continue
                        cur.execute("""
                            SELECT "Id", "UserEmbeddingVector"
                            FROM "User"
                            WHERE "DomainId" = %s AND "Id" = ANY(%s::int[])
                        """, (self.domain_id, chunk,))
                        all_rows_users.extend(cur.fetchall())
        except Exception as e:
            print(f"❌ Error querying user chunks: {e}")
            raise

        self.user_emb_dict = {}
        processed_user_ids = set()
        k_dim_for_user = k_bert_dim
        for user_id, emb_vec in all_rows_users:
            user_id = str(user_id)
            if emb_vec is not None:
                try:
                    loaded_emb = np.array(emb_vec, dtype=float)
                    if len(loaded_emb) == k_dim_for_user:
                        self.user_emb_dict[user_id] = loaded_emb
                    else:
                        self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
                except Exception as e:
                    self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            else:
                self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)
            processed_user_ids.add(user_id)

        users_not_in_db_table = set(users_to_load) - processed_user_ids
        if users_not_in_db_table:
            for user_id in users_not_in_db_table:
                self.user_emb_dict[user_id] = np.zeros(k_dim_for_user)

    # THAY ĐỔI: Gộp query vào 1 transaction
    def load_model_from_db(self, model_id):
        item_factors = {}
        item_biases = {}
        temp_k_from_factors = None
        user_factors = {}
        user_biases = {}
        mu_val = 3.5

        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT "ItemId", "ItemBias", "ItemFactors"
                        FROM "ItemFactor"
                        WHERE "ModelId" = %s
                        ORDER BY "ItemId"
                    """, (model_id,))
                    rows_items = cur.fetchall()
                    for item_id, bias, factors in rows_items:
                        item_id = str(item_id)
                        if item_id not in self.item2idx: continue
                        if bias is not None and factors is not None:
                            item_biases[item_id] = float(bias)
                            item_factors[item_id] = np.array(factors, dtype=float)
                            if temp_k_from_factors is None:
                                temp_k_from_factors = len(item_factors[item_id])

                    cur.execute("""
                        SELECT "UserId", "UserBias", "UserFactors"
                        FROM "UserFactor"
                        WHERE "ModelId" = %s
                        ORDER BY "UserId"
                    """, (model_id,))
                    rows_users = cur.fetchall()
                    for user_id, bias, factors in rows_users:
                        user_id = str(user_id)
                        if user_id not in self.user2idx: continue
                        if bias is not None and factors is not None:
                            user_biases[user_id] = float(bias)
                            user_factors[user_id] = np.array(factors, dtype=float)
                            if self.k is None and len(user_factors[user_id]) > 0:
                                self.k = len(user_factors[user_id])

                    cur.execute('SELECT "AverageRating" FROM "Model" WHERE "Id"=%s', (model_id,))
                    mu_row = cur.fetchone()
                    if mu_row and mu_row[0] is not None:
                        mu_val = float(mu_row[0])
        except Exception as e:
            print(f"❌ Error loading model state from DB: {e}")
            raise

        if temp_k_from_factors is not None:
            self.k = temp_k_from_factors
            print(f"   Inferred K={self.k} from loaded item factors.")
        elif self.k is None:
            print("⚠️ Warning: Cannot determine K from loaded factors. Using default.")

        self.mu = mu_val
        n_users_loaded = len(self.users)
        n_items_loaded = len(self.items_list)
        
        P_init_arr = np.zeros((n_users_loaded, self.k), dtype=float)
        Q_init_arr = np.zeros((n_items_loaded, self.k), dtype=float)
        b_u_init_arr = np.zeros(n_users_loaded, dtype=float)
        b_i_init_arr = np.zeros(n_items_loaded, dtype=float)
        
        for user_id, idx in self.user2idx.items():
            if user_id in user_factors:
                P_init_arr[idx] = user_factors[user_id]
            if user_id in user_biases:
                b_u_init_arr[idx] = user_biases[user_id]

        for item_id, idx in self.item2idx.items():
            if item_id in item_factors:
                Q_init_arr[idx] = item_factors[item_id]
            if item_id in item_biases:
                b_i_init_arr[idx] = item_biases[item_id]
                
        self.model = self.UEIEModel(n_users_loaded, n_items_loaded, self.k, self.mu).to(self.device)
        self.model.eval()

        print(f"✅ Loaded Factors/Biases into PyTorch Model for Prediction.")

    def load_bert(self):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.bert_model.eval()
            self.bert_model.to(self.device)
            print("BERT model loaded.")
            return self.tokenizer, self.bert_model
        except Exception as e:
            print(f"❌ Error loading BERT model: {e}")
            self.tokenizer, self.bert_model = None, None
            raise

    def get_embedding(self, text):
        if self.tokenizer is None or self.bert_model is None:
            return np.zeros(768)

        if not text or not isinstance(text, str):
            k_bert_dim = self.bert_model.config.dim
            return np.zeros(k_bert_dim)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = self.bert_model(**inputs)
            except Exception as e:
                k_bert_dim = self.bert_model.config.dim
                return np.zeros(k_bert_dim)

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        embedding = mean_pooled.squeeze().cpu().numpy()
        return embedding

    def compute_inferred_preferences(self):
        if not self.ratings:
            print("⚠️ No ratings loaded. Cannot compute z_ui. Returning zero array.")
            return np.array([])
            
        if not self.user_emb_dict or not self.item_emb_dict:
            print("⚠️ User or Item embedding dict is empty. Using zeros for z_ui.")
            return np.zeros(len(self.ratings))

        inferred = np.zeros(len(self.ratings))
        errors = 0
        k_dim_check = -1

        try:
            k_dim_check = len(next(iter(self.item_emb_dict.values())))
        except StopIteration:
            print("⚠️ Item embedding dict is empty after load! Using zeros for z_ui.")
            return np.zeros(len(self.ratings))

        for idx, (u, i, _) in enumerate(tqdm(self.ratings, desc="   Calculating z_ui")):
            user_vec = self.user_emb_dict.get(u)
            item_vec = self.item_emb_dict.get(i)

            valid = True
            if user_vec is None or user_vec.shape[0] != k_dim_check:
                valid = False
            if item_vec is None or item_vec.shape[0] != k_dim_check:
                valid = False

            if valid:
                try:
                    inferred[idx] = np.dot(user_vec, item_vec)
                except ValueError as e:
                    inferred[idx] = 0.0
                    errors += 1
            else:
                inferred[idx] = 0.0
                errors += 1

        if errors > 0:
            print(f"⚠️ Encountered {errors} issues (missing/mismatched embeddings) during z_ui calculation.")

        if np.isnan(inferred).any() or np.isinf(inferred).any():
            print("⚠️ Inferred preferences contain NaN/Inf. Replacing with 0 before scaling.")
            inferred = np.nan_to_num(inferred, nan=0.0, posinf=0.0, neginf=0.0)

        if np.ptp(inferred) > 1e-9:
            scaler = MinMaxScaler(feature_range=(1, 5))
            try:
                scaled_inferred = scaler.fit_transform(inferred.reshape(-1, 1)).flatten()
                return scaled_inferred
            except ValueError as e:
                print(f"⚠️ Error scaling z_ui: {e}. Using clipped unscaled values.")
                return np.clip(inferred, 1.0, 5.0)
        else:
            mu_val = self.mu if self.mu != 0.0 else 3.5
            print(f"⚠️ Inferred preferences have no variance. Scaling skipped. Using constant value: {mu_val:.2f}")
            return np.full(len(self.ratings), mu_val)

    def train_model(self, epochs=500, batch_size=256):
        if not self.ratings: print("⚠️ No training data. Skipping training."); return
        if self.model is None: print("⚠️ Model not initialized. Skipping."); return
        if self.optimizer is None: print("⚠️ Optimizer not initialized. Skipping."); return

        print(f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}, device={self.device}")
        self.model.train()
        n = len(self.ratings)
        prev_avg_loss = float('inf')
        tolerance = 1e-6

        try:
            user_indices = [self.user2idx[u] for u, _, _ in self.ratings]
            item_indices = [self.item2idx[i] for _, i, _ in self.ratings]
            user_idx_all = torch.tensor(user_indices, device=self.device, dtype=torch.long)
            item_idx_all = torch.tensor(item_indices, device=self.device, dtype=torch.long)
            r_all = torch.tensor([r for _, _, r in self.ratings], dtype=torch.float32, device=self.device)
        except KeyError as e:
            print(f"❌ ERROR preparing tensors: User or Item ID '{e}' missing from index map! Aborting training.")
            return

        if self.inferred_prefs is None or len(self.inferred_prefs) != n:
            z_all = torch.full((n,), self.mu, dtype=torch.float32, device=self.device)
        else:
            z_all = torch.tensor(self.inferred_prefs, dtype=torch.float32, device=self.device)

        for epoch in range(epochs):
            total_main_loss_sum = 0.0
            perm = torch.randperm(n, device=self.device)
            user_idx_shuffled = user_idx_all[perm]
            item_idx_shuffled = item_idx_all[perm]
            r_shuffled = r_all[perm]
            z_shuffled = z_all[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if start == end: continue
                
                u_batch = user_idx_shuffled[start:end]
                i_batch = item_idx_shuffled[start:end]
                r_batch = r_shuffled[start:end]
                z_batch = z_shuffled[start:end]
                current_batch_size = len(u_batch)
                
                r_hat = self.model(u_batch, i_batch)
                err_r = r_batch - r_hat
                err_z = z_batch - r_hat
                main_loss_batch = 0.5 * (err_r**2 + self.weight * err_z**2)
                main_loss_mean = main_loss_batch.mean()

                loss = main_loss_mean
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_main_loss_sum += main_loss_mean.item() * current_batch_size

            avg_main_loss = total_main_loss_sum / n
            with torch.no_grad():
                reg_p = torch.sum(self.model.P.weight**2)
                reg_q = torch.sum(self.model.Q.weight**2)
                reg_bu = torch.sum(self.model.b_u.weight**2)
                reg_bi = torch.sum(self.model.b_i.weight**2)
            
            final_reg_loss = 0.5 * self.lam * (reg_p + reg_q + reg_bu + reg_bi).item()
            avg_total_loss_to_track = avg_main_loss + final_reg_loss / n

            if abs(prev_avg_loss - avg_total_loss_to_track) < tolerance:
                print(f"✅ Early stopping at epoch {epoch+1} (|Δloss|={abs(prev_avg_loss - avg_total_loss_to_track):.6e} < {tolerance})")
                break
            prev_avg_loss = avg_total_loss_to_track

        print("🏁 Training finished.")

    # THAY ĐỔI: Tự mở/đóng kết nối, xử lý lỗi riêng cho từng phần để tránh crash toàn bộ
    def write_model_to_db(self):
        if self.train_mode != 'train': print("Not in train mode. Skipping save."); return
        if self.db_config is None: raise ValueError("❌ DB config required.")
        if self.model is None: print("⚠️ Model not trained. Cannot save."); return
        
        self.model.eval()
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    if self.user_emb_dict and self.item_emb_dict:
                        update_count_u, update_count_i = 0, 0
                        try:
                            for user_id in self.users:
                                if user_id in self.user_emb_dict:
                                    emb = self.user_emb_dict[user_id]
                                    cur.execute('UPDATE "User" SET "UserEmbeddingVector"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s', (emb.tolist(), user_id))
                                    update_count_u += cur.rowcount
                            for item_id in self.items_list:
                                if item_id in self.item_emb_dict:
                                    emb = self.item_emb_dict[item_id]
                                    cur.execute('UPDATE "Item" SET "EmbeddingVector"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s', (emb.tolist(), item_id))
                                    update_count_i += cur.rowcount
                            print(f"Updated embeddings for {update_count_u} users, {update_count_i} items.")
                            conn.commit()
                        except Exception as e:
                            print(f"❌ Error updating embeddings: {e}")
                            conn.rollback()

                    with torch.no_grad():
                        all_user_biases = self.model.b_u.weight.squeeze().cpu().numpy()
                        all_user_factors = self.model.P.weight.cpu().numpy()
                        all_item_biases = self.model.b_i.weight.squeeze().cpu().numpy()
                        all_item_factors = self.model.Q.weight.cpu().numpy()

                    user_factor_data = []
                    item_factor_data = []
                    
                    users_in_train = set(u for u, _, _ in self.ratings) if self.ratings else set()
                    items_in_train = set(i for _, i, _ in self.ratings) if self.ratings else set()

                    for user_id in self.users:
                        if user_id in users_in_train:
                            idx = self.user2idx[user_id]
                            user_factor_data.append((
                                user_id, float(all_user_biases[idx]), all_user_factors[idx].tolist(), self.model_id
                            ))
                        else:
                            idx = self.user2idx.get(user_id)
                            if idx is not None:
                                user_factor_data.append((
                                    user_id, 0.0, all_user_factors[idx].tolist(), self.model_id
                                ))
                    
                    for item_id in self.items_list:
                        if item_id in items_in_train:
                            idx = self.item2idx[item_id]
                            item_factor_data.append((
                                item_id, float(all_item_biases[idx]), all_item_factors[idx].tolist(), self.model_id
                            ))
                        else:
                            idx = self.item2idx.get(item_id)
                            if idx is not None:
                                item_factor_data.append((
                                    item_id, 0.0, all_item_factors[idx].tolist(), self.model_id
                                ))

                    if user_factor_data:
                        sql_user = """
                        INSERT INTO "UserFactor" ("UserId", "UserBias", "UserFactors", "ModelId")
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT ("UserId", "ModelId") DO UPDATE SET 
                            "UserBias" = EXCLUDED."UserBias",
                            "UserFactors" = EXCLUDED."UserFactors";
                        """
                        psycopg2.extras.execute_batch(cur, sql_user, user_factor_data)

                    if item_factor_data:
                        sql_item = """
                        INSERT INTO "ItemFactor" ("ItemId", "ItemBias", "ItemFactors", "ModelId")
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT ("ItemId", "ModelId") DO UPDATE SET
                            "ItemBias" = EXCLUDED."ItemBias",
                            "ItemFactors" = EXCLUDED."ItemFactors";
                        """
                        psycopg2.extras.execute_batch(cur, sql_item, item_factor_data)
                    
                    cur.execute("""
                        UPDATE "Model" SET "AverageRating"=%s, "ModifiedAt"=NOW() WHERE "Id"=%s 
                    """, (float(self.mu), self.model_id))
                    
                    conn.commit()
                    print(f"Saved model factors. Users: {len(self.user2idx)}, Items: {len(self.item2idx)}")

        except Exception as e:
            print(f"   ❌ Error saving factors/biases (UPSERT stage): {e}")
    
    def predict(self, user, item, p):
        if self.model is None:
            return self.mu if self.mu else 3.5

        self.model.eval()
        with torch.no_grad():
            u_idx_val = self.user2idx.get(user)
            i_idx_val = self.item2idx.get(item)
            
            if u_idx_val is None:
                if i_idx_val is not None:
                    i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
                    b_i = self.model.b_i(i_idx).squeeze().item()
                    prediction = self.mu + b_i
                else:
                    prediction = self.mu
                return np.clip(prediction, 1.0, 5.0)

            if i_idx_val is None:
                u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
                b_u = self.model.b_u(u_idx).squeeze().item()
                prediction = self.mu + b_u
                return np.clip(prediction, 1.0, 5.0)

            u_idx = torch.tensor([u_idx_val], device=self.device, dtype=torch.long)
            i_idx = torch.tensor([i_idx_val], device=self.device, dtype=torch.long)
            prediction = self.model(u_idx, i_idx).item()
            
            return np.clip(prediction, 1.0, 5.0)